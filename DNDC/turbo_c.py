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
import traceback
from dataclasses import dataclass
from copy import deepcopy
import sys

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective, IdentityMCObjective
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from botorch.test_functions import Ackley, Rosenbrock

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("always", category=BadInitialCandidatesWarning)
warnings.filterwarnings("always", category=RuntimeWarning)
warnings.simplefilter("always", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

SMOKE_TEST = False

# specifies weights of each function
weights = torch.tensor([1.0, 1.0, 1.0, 1.0])

@dataclass
class cTurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7 # Note: Default is 0.5**7 from Turbo paper, sometimes change to 0.5**8 to give additional chance to shrink
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

class cTurbo1:
    def __init__(
        self,
        g,                      # black-box functions, evaluates inputs x (b x d) to y (b x o)
        weights,                # weights for each function f(x) = weights'*g(x)
        dim,                    # problem dimension
        n_init,                 # number of initial data points (recommended is 2*d)
        max_evals,              # total number of function evaluations
        batch_size=1,           # number of points in the batch
        verbose=True,           # prints results if true
        max_data_length=1000,   # maximum amount of data to include when training GPs
        max_cholesky_size=float("inf"), # maximum size of Cholesky decomposition
        max_attempts=5,         # number of restarts in gp training
        acqf='ts',              # acquisition function type, only supports TS and (log)EI
        n_candidates=None,      # number of candidates for Thompson sampling
        num_restarts=10,        # number of restarts for optimizing acquisition function (only for logEI)
        raw_samples=512,        # number of raw samples to evaluate acquisition at before optimization (only for logEI)
        dtype=torch.double,     # data type
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # device
        seed=0,                 # random seed
    ):

        # simple input checks
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)

        # save function information
        self.g = g
        self.weights = weights
        self.dim = dim
        self.o = len(weights)

        # settings of algorithm
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_data_length = max_data_length
        self.acqf = acqf
        self.n_candidates = n_candidates
        self.max_cholesky_size = max_cholesky_size
        self.max_attempts = max_attempts
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

        # save the full history of measurements
        self.X = torch.zeros((0, self.dim))
        self.gX = torch.zeros((0, self.o))
        self.fX = torch.zeros((0, 1))

        # initialize the state of cTurbo and other relevant parameters
        self.restart()
        self.n_evals = 0
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.first_run = True

    def obj(self, samples, X=None):
        # this is a helper to make the weights'*samples evaluation easier
        weights = self.weights
        if samples.dim() == 2:
            samples = samples.unsqueeze(1)
        return torch.sum(weights * samples, dim=-1)

    def restart(self):
        self.state = []
        self._X = []
        self._gX = []
        self._fX = []
        self.prev_model = None
        self.prev_X = None
        self.prev_gX = None
        self.prev_fX = None

    def initialize_state(self, Y_best):
        self.state = cTurboState(dim=self.dim, batch_size=self.batch_size, best_value=Y_best)

    def update_state(self, Y_next):
        state = self.state
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
        self.state = state

    def get_initial_points(self, n_pts, seed=None):
        if seed is None:
            seed = torch.randint(0,int(1e6),()).item()
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)
        return X_init

    def train_gp(self, X, Z, Y):
        try:
            models = []
            for i, train_Z in enumerate(Z.T):
                train_Z = train_Z.reshape((-1, 1))
                train_Z = (train_Z - train_Z.mean()) / train_Z.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 0.5))
                covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0)))  # Use the same lengthscale prior as in the TuRBO paper
                  # NOTE: previously used RBF: covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))  # Use the same lengthscale prior as in the TuRBO paper
                model = SingleTaskGP(X, train_Z, covar_module=covar_module, likelihood=likelihood)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    fit_gpytorch_mll(mll, max_attempts=self.max_attempts)
                models.append(model)
            # wrap the list of models into a ModelListGP
            model = ModelListGP(*models)
            # store the trained models for future use (in case of failure)
            self.prev_model = deepcopy(model)
            self.prev_X = deepcopy(X)
            self.prev_gX = deepcopy(Z)
            self.prev_fX = deepcopy(Y)
        except:
            print(f"    Model training failed, so reusing previous model")
            model = self.prev_model
            X = self.prev_X
            Z = self.prev_gX
            Y = self.prev_fX
        return model, X, Z, Y

    def generate_batch(self, model, X, gX, fX):
        # extract key parameters and run simple checks
        state = self.state
        acqf = self.acqf
        n_candidates = self.n_candidates
        Z = gX
        Y = fX
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # calculate the effective lengthscales from the composite model
        dim = X.shape[-1]
        train_stds = Z.std(dim=0)
        vars = torch.tensor([model.models[i].covar_module.outputscale.detach() for i in range(len(model.models))]) # estimated outputscale variance from each individual kernel
        scale_factors = train_stds**2 * vars / torch.sum(train_stds**2 * vars) # scaled total variance based on the scaling factors
        length_scales = [model.models[i].covar_module.base_kernel.lengthscale.squeeze().detach() for i in range(len(model.models))] # lengthscales for all models
        length_scales = torch.stack(length_scales).T
        weights = torch.sum(scale_factors.repeat((dim,1)) * (1.0 / length_scales**2), dim=1)**(-1/2) # use simple rule to calculate effective lengthscale

        # scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
            # OLD code: weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        # TODO: need to add back composite form of logEI
        if acqf == "ts":
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # create a transformation objective for taking Z to Y, need to account for scaling used to train GPs
            def obj_transform(samples, X): # need to "untransform" the samples generated by the GPs
                return self.obj(samples * Z.std(dim=0) + Z.mean(dim=0)) # can delete the Z.mean(dim=0) part since the constant should not shift objective value
            mc_objective = GenericMCObjective(obj_transform)

            # sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, objective=mc_objective, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=self.batch_size)

        # added commposite form of EI using ScalarizedPosteriorTransform (see bottom of https://botorch.org/docs/tutorials/custom_acquisition/)
        elif acqf == "ei":
            scaled_weights = self.weights * Z.std(dim=0)
            offset = torch.sum(self.weights * Z.mean(dim=0))
            pt = ScalarizedPosteriorTransform(scaled_weights, offset)
            if self.batch_size == 1:
                log_ei = LogExpectedImprovement(model, Y.max(), posterior_transform=pt)
            else:
                log_ei = qLogExpectedImprovement(model, Y.max(), posterior_transform=pt)
            X_next, acq_value = optimize_acqf(
                log_ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
        return X_next

    def optimize(self):
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.max().item()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # initialize parameters
            self.restart()

            # generate and evaluation initial design
            if self.first_run:
                X_init = self.get_initial_points(self.n_init, self.seed)
                self.first_run = False
            else:
                X_init = self.get_initial_points(self.n_init)
            gX_init = None
            for x in X_init:
                if gX_init is None:
                    gX_init = self.g(x).unsqueeze(0)
                else:
                    gX_init = torch.cat([gX_init, self.g(x).unsqueeze(0)], dim=0)
            fX_init = self.obj(gX_init)

            # update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._gX = deepcopy(gX_init)
            self._fX = deepcopy(fX_init)

            # initialize state of algorithm
            self.initialize_state(max(fX_init).item())

            if self.verbose:
                fbest = self._fX.max().item()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            # append data to global history
            self.X = torch.cat((self.X, deepcopy(X_init)), dim=0)
            self.gX = torch.cat((self.gX, deepcopy(gX_init)), dim=0)
            self.fX = torch.cat((self.fX, deepcopy(fX_init)), dim=0)

            # loop over updated trust region until convergence
            while self.n_evals < self.max_evals and not self.state.restart_triggered:
                # fit the models
                model, X, gX, fX = self.train_gp(deepcopy(self._X)[-self.max_data_length:], deepcopy(self._gX)[-self.max_data_length:], deepcopy(self._fX)[-self.max_data_length:])

                # run acquisition function optimization inside the Cholesky context
                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    # create a batch
                    X_next = self.generate_batch(model, X, gX, fX)

                # evaluate the batch
                gX_next = None
                for x in X_next:
                    if gX_next is None:
                        gX_next = self.g(x).unsqueeze(0)
                    else:
                        gX_next = torch.cat([gX_next, self.g(x).unsqueeze(0)], dim=0)
                fX_next = self.obj(gX_next)

                # update state, budget, and append data
                self.update_state(Y_next=fX_next)
                self.n_evals += self.batch_size
                self._X = torch.cat((self._X, X_next), dim=0)
                self._gX = torch.cat((self._gX, gX_next), dim=0)
                self._fX = torch.cat((self._fX, fX_next), dim=0)

                if self.verbose:
                    print(f"{self.n_evals}) Best value: {self.state.best_value:.2e}, TR length: {self.state.length:.2e}")
                    sys.stdout.flush()
                # if self.verbose and fX_next.max().item() > self.fX.max().item():
                #     n_evals, fbest = self.n_evals, fX_next.max().item()
                #     print(f"{n_evals}) New best: {fbest:.4}")
                #     sys.stdout.flush()

                # append data to global history
                self.X = torch.cat((self.X, deepcopy(X_next)), dim=0)
                self.gX = torch.cat((self.gX, deepcopy(gX_next)), dim=0)
                self.fX = torch.cat((self.fX, deepcopy(fX_next)), dim=0)


if __name__ == ' __main__':             
    
    from calib_objective import ObjFunc
    from DNDCrun import DNDC
    
    simulator = DNDC()
    a = ObjFunc()
    dim = 34

    def g(x):
        """This is a helper function we use to unnormalize and evaluate a point"""
        # print(x)
        return -a(x)
    
    # set initial parameters
    batch_size = 4
    n_init = 2 * dim
    max_evals = 2000 if not SMOKE_TEST else 10
    max_cholesky_size = float("inf")  # Always use Cholesky
    
    cturbo1 = cTurbo1(g, weights, dim, n_init, max_evals, batch_size, max_attempts=3, acqf='ts', max_data_length=500, n_candidates=2000, num_restarts=5, raw_samples=2056, seed=0)
    
    cturbo1.optimize()
    
    Y_cturbo = cturbo1.fX
    Y_opt_cturbo = Y_cturbo.max().item()
    X_opt_cturbo = cturbo1.X[torch.argmax(cturbo1.fX)].detach().numpy()
    print("Best-found input:", X_opt_cturbo)
    print("Best-found objective value:", Y_opt_cturbo)