# -*- coding: utf-8 -*-
"""
Composite TuRBO Optimization Algorithm

Adapated from method of Eriksson et al., citation:
    
@inproceedings{eriksson2019scalable,
  title = {Scalable Global Optimization via Local {Bayesian} Optimization},
  author = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D and Poloczek, Matthias},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {5496--5507},
  year = {2019},
  url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf},
}

@author: kdonn
"""

import os
import math
import warnings
from dataclasses import dataclass

import torch
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective, IdentityMCObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP, ModelListGP
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from botorch.test_functions import Ackley, Rosenbrock

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from calib_objective import simulator_run, objective_function
from DNDCrun import DNDC

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

SMOKE_TEST = False

# Function Definition:

simulator = DNDC()

dim = 34

# # first function
# fun_1, fun_2, fun_3, fun_4 = simulator_run(x)
# fun_1.bounds[0, :].fill_(0)
# fun_1.bounds[1, :].fill_(1)
# lb, ub = fun_1.bounds

# # Assume same bounds for both functions
# fun_2 = simulator_run[1]
# fun_3 = simulator_run[2]
# fun_4 = simuator_run[3]

# set initial parameters
batch_size = 4
n_init = dim
max_cholesky_size = float("inf")  # Always use Cholesky

# this performs evaluation of individual functions
def eval_functions(x):
    """This is a helper function we use to unnormalize and evaluate a point"""
    return -simulator_run(x)

# this is the overall objective function
def sum_of_funcs(samples, X=None):
    if samples.dim() == 2:
        samples = samples.unsqueeze(1)
    return torch.sum(samples, dim=-1)

# function to get initial points for evaluation
def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

@dataclass
class CompositeTurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")   # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10             # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

state = CompositeTurboState(dim=dim, batch_size=batch_size)
# print(state)

def generate_batch(
    state,
    model,  # GP model class (in our case will be a list of SingleTaskGP models)
    X,  # Evaluated points on the domain [0, 1]^d
    Z,  # Intermediate function values
    Y,  # Objective function values (this is total objective function)
    batch_size,
    obj=None,  # Objective transformation
    n_candidates=None,  # Number of candidates for Thompson sampling
    acqf="ts",  # only support "ts" for now
):
    assert acqf in ("ts")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Calculate the effective lengthscales from the composite model
    dim = X.shape[-1]
    train_stds = Z.std(dim=0)
    vars = torch.tensor([model.models[i].covar_module.outputscale.detach() for i in range(len(model.models))]) # estimated outputscale variance from each individual kernel
    scale_factors = train_stds**2 * vars / torch.sum(train_stds**2 * vars) # scaled total variance based on the scaling factors
    length_scales = [model.models[i].covar_module.base_kernel.lengthscale.squeeze().detach() for i in range(len(model.models))] # lengthscales for all models
    length_scales = torch.stack(length_scales).T
    weights = torch.sum(scale_factors.repeat((dim,1)) * (1.0 / length_scales**2), dim=1)**(-1/2) # use simple rule to calculate effective lengthscale

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
        # OLD code: weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Create a transformation objective for taking Z to Y, need to account for scaling used to train GPs
        if obj is None:
            mc_objective = IdentityMCObjective()
        else:
            def obj_transform(samples, X): # need to "untransform" the samples generated by the GPs
                return obj(samples * Z.std(dim=0) + Z.mean(dim=0))
            mc_objective = GenericMCObjective(obj_transform)

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, objective=mc_objective, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

X_cturbo = get_initial_points(dim, n_init)
Z_cturbo = None
for x in X_cturbo:
    if Z_cturbo is None:
        Z_cturbo = eval_functions(x).unsqueeze(0)
    else:
        Z_cturbo = torch.cat([Z_cturbo, eval_functions(x).unsqueeze(0)], dim=0)
Y_cturbo = sum_of_funcs(Z_cturbo)

state = CompositeTurboState(dim, batch_size=batch_size, best_value=max(Y_cturbo).item())

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

torch.manual_seed(0)

while not state.restart_triggered:  # Run until Composite TuRBO converges
    # Fit GP model for all outputs
    models = []
    for train_Z in Z_cturbo.T:
        train_Z = train_Z.reshape((-1,1))
        train_Z = (train_Z - train_Z.mean()) / train_Z.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel( RBFKernel(ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)) ) # Use the same lengthscale prior as in the TuRBO paper
        model = SingleTaskGP(X_cturbo, train_Z, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        models.append(model)
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)
    model = ModelListGP(*models)

    # Do the acquisition function optimization inside the Cholesky context
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        # Create a batch
        X_next = generate_batch(
            state=state,
            model=model,
            X=X_cturbo,
            Z=Z_cturbo,
            Y=Y_cturbo,
            batch_size=batch_size,
            obj=sum_of_funcs,
            n_candidates=N_CANDIDATES,
            acqf="ts",
        )

    Z_next = None
    for x in X_next:
        if Z_next is None:
            Z_next = eval_functions(x).unsqueeze(0)
        else:
            Z_next = torch.cat([Z_next, eval_functions(x).unsqueeze(0)], dim=0)
    Y_next = sum_of_funcs(Z_next)

    # Update state
    state = update_state(state=state, Y_next=Y_next)

    # Append data
    X_cturbo = torch.cat((X_cturbo, X_next), dim=0)
    Z_cturbo = torch.cat((Z_cturbo, Z_next), dim=0)
    Y_cturbo = torch.cat((Y_cturbo, Y_next), dim=0)

    # Print current status
    print(f"{len(X_cturbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
    
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

%matplotlib inline

names = ["C-TuRBO-1"]
runs = [Y_cturbo]
fig, ax = plt.subplots(figsize=(8, 6))

for name, run in zip(names, runs):
    fx = np.maximum.accumulate(run.cpu())
    plt.plot(-fx, marker="", lw=3)

len_x = torch.tensor([0])

plt.plot(len_x.repeat(len(fx)), "k--", lw=3)
plt.ylabel("Loss", fontsize=18)
plt.xlabel("Evaluations", fontsize=18)
#plt.title("20D Ackley + const * 20D Rosenb", fontsize=24)
plt.xlim([0, len(Y_cturbo)])
#plt.ylim([-15, 1])
ax.set_yscale('log')

plt.grid(True)
plt.tight_layout()
#plt.legend(
#    names + ["Global optimal value"],
#    loc="lower center",
#    bbox_to_anchor=(0, -0.08, 1, 1),
#    bbox_transform=plt.gcf().transFigure,
#    ncol=5,
#    fontsize=16,
#)
plt.show()

import pandas as pd
df_Y_cturbo = Y_cturbo
df_X_cturbo = X_cturbo
df_Z_cturbo = Z_cturbo
df_fx = fx
df_Y_cturbo = pd.DataFrame(df_Y_cturbo)
df_X_cturbo = pd.DataFrame(df_X_cturbo)
df_Z_cturbo = pd.DataFrame(df_Z_cturbo)
df_fx = pd.DataFrame(df_fx)
df_X_cturbo.to_csv('df_X_cturbo.csv', sep=',', index = False, encoding='utf-8')
df_Y_cturbo.to_csv('df_Y_cturbo.csv', sep=',', index = False, encoding='utf-8')
df_Z_cturbo.to_csv('df_Z_cturbo.csv', sep=',', index = False, encoding='utf-8')
df_fx.to_csv('df_fx.csv', sep=',', index = False, encoding='utf-8')