# -*- coding: utf-8 -*-
"""
Hessian Estimation
@author: donnelly.235
"""

import os
import math
import warnings
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from calib_objective import simulator_run, objective_function
from DNDCrun import DNDC
# from TuRBO_algo import eval_objective

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# import numpy as np
import pandas as pd
import pickle

dim = 34

# Specify TuRBO data to estimate Hessian around

# with open('gp_model4.pkl', 'rb') as f:
#     model = pickle.load(f)

with open('X_turbo.pkl','rb') as f:
    X_turbo = pickle.load(f)

with open('Y_turbo.pkl','rb') as f:
    Y_turbo = pickle.load(f)
    
with open('model.pkl','rb') as f:
    model = pickle.load(f)

# X_turbo = torch.tensor(pd.read_csv('df_X_TuRBO_4.csv').values)
# Y_turbo = torch.tensor(pd.read_csv('df_Y_TuRBO_4.csv').values)

def GP_func(x):
    return model.posterior(x).mean

def Hessian_calc(model, X_eval):
    return torch.autograd.functional.hessian(GP_func, X_eval)

def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    sr1, sr2, sr3, sr4 = simulator_run(x)
    return -objective_function(sr1,sr2,sr3,sr4)

X_MLE = X_turbo[torch.argmax(Y_turbo)].unsqueeze(0) # X that results in the largest negative loss
GP_Hessian_prev = Hessian_calc(model, X_MLE) # Calculate the Hessian at X_MLE
# Define a small hypercube around x_MLE for sampling extra points (feel free to play around with this hyperparameters)
LB = X_MLE - 0.01
UB = X_MLE + 0.01
np = 10 # number of point to select per iteration (feel free to play around with this hyperparameters)
dist = 10
iterations = 0

while dist>1: # escape the loop if the distance between current Hessian and the previous Hessian is smaller than 0.1 (feel free to play around with this hyperparameters)    
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_next = LB+(UB-LB)*sobol.draw(np).to(dtype=dtype, device=device) # randomly sample np points around X_MLE
    X_next = torch.clamp(X_next, min=0, max=1) # make sure the input is between zero and one
    Y_next = torch.tensor(
        [eval_objective(x) for x in X_next], dtype=dtype, device=device
    ).unsqueeze(-1)
    X_turbo = torch.cat((X_turbo, X_next), dim=0)
    Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

    # Re-fit the GP
    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(
            nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
        )
    )
    model = SingleTaskGP(
        X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model) 
    fit_gpytorch_mll(mll)

    # Calculate the new Hessian
    GP_Hessian_new = Hessian_calc(model, X_MLE)

    # Calculate the distance between the new Hessian and the previous Hessian
    #dist = torch.sum((GP_Hessian_new - GP_Hessian_prev)**2)
    dist = torch.norm(GP_Hessian_new - GP_Hessian_prev, p='fro')

    GP_Hessian_prev = GP_Hessian_new.clone()
    print(dist)
    iterations += 1
    
est_GP_Hessian = GP_Hessian_prev.view(34,34)
cov_matrix = torch.linalg.inv(est_GP_Hessian)
standard_errors = torch.sqrt(torch.diag(cov_matrix))
z_score = 1.96
lower = X_MLE - z_score*standard_errors
upper = X_MLE + z_score*standard_errors
CIs = torch.stack((lower,upper),dim=1)

# Convert tensors to numpy for easier plotting
cov_matrix_np = cov_matrix.numpy()

# Eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_np)

axes_lengths = np.sqrt(eigenvalues) * z_score
   
# Generate ellipse points
theta = np.linspace(0, 2 * np.pi, 100)
ellipse = np.array([
    axes_lengths[0] * np.cos(theta),
    axes_lengths[1] * np.sin(theta)
])


#num_samples = 100
#samples = torch.distributions.MultivariateNormal(X_MLE, covariance_matrix=cov_matrix)

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10,6))
plt.scatter(X_turbo[:, 0], X_turbo[:, 0], alpha=0.3, label='Sample Points')

font_axis_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 22,
        }
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16


# Plot confidence intervals as error bars
z_score = 1.96  # 95% confidence
plt.errorbar(X_turbo[0], X_turbo[0], 
             xerr=z_score * torch.sqrt(cov_matrix[0, 0]), 
             yerr=z_score * torch.sqrt(cov_matrix[1, 1]), 
             fmt='o', color='red', label='MLE Estimate (95% CI)')

plt.xlabel('Clay_fraction',fontdict=font_axis_publish)
plt.ylabel('Clay_fraction',fontdict=font_axis_publish)
plt.legend()
plt.grid(True)
plt.show()