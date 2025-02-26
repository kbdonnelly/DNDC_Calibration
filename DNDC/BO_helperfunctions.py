# -*- coding: utf-8 -*-
"""
@author: donnelly.235
"""

import pdb
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import math
import gpytorch
import botorch
import torch
torch.set_default_dtype(torch.float64)
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, LinearKernel, PeriodicKernel
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, PosteriorMean
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf

from scipy import optimize
from sklearn.metrics import r2_score
# from skquant.opt import minimize as skqmin

# turn off warnings for clean output
import warnings
warnings.filterwarnings('ignore')

font_axis_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 22,
        }
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16

def GP_training(X, Y, kernel_type, noise_free=False, plot_1d=False, plot_bounds=None):
    """
      -----------
    Arg:
    X: Features/ Input vector -- torch tensor 
    Y: Mapping/ Target variable vector -- torch tensor
    kernel_type: 'RBF'/'Linear'/'Periodic'/'Matern05'/'Matern15'/'Matern25' select one -- str
    noise_free: True or False
    plot_1d: True or False
    plot_bounds: Tuple of lower and upper bounds (xL, xU)

    ----------
    returns:

    model: a GP model object in train mode -- gpytorch
    """  

    # make sure input data is shaped properly (ntrain by ninputs)
    if X.ndim < 2:
      print("Need to specify as matrix of size ntrain by ninputs")

    # make sure training data has the right dimension
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)

    # output scaling
    standardize = Standardize(m=Y.shape[-1])
    outcome_transform = standardize

    # select covariance module
    input_dim = X.shape[-1]
    if kernel_type == 'RBF':
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
    elif kernel_type == 'Linear':
        covar_module = ScaleKernel(LinearKernel(ard_num_dims=input_dim))
    elif kernel_type == 'Periodic':
        covar_module = ScaleKernel(PeriodicKernel(ard_num_dims=input_dim))
    elif kernel_type == 'Matern05':
        covar_module = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=input_dim))
    elif kernel_type == 'Matern15':
        covar_module = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=input_dim))
    elif kernel_type == 'Matern25':
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=input_dim))

    # set the likelihood
    if noise_free:
      likelihood = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-5, upper_bound=1e-3))
    else:
      likelihood = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-5, upper_bound=100))

    # define the model
    model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=covar_module, likelihood=likelihood, outcome_transform=outcome_transform)

    # call the training procedure
    model.outcome_transform.eval()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # put in eval mode
    model.eval()

    # plot if specified 
    if plot_1d:
      if input_dim == 1:
        if plot_bounds is None:
          print("Plot bounds not specified!")
        else:
          x_list = torch.linspace(plot_bounds[0], plot_bounds[1], 101)
          preds = model.posterior(x_list.unsqueeze(-1))
          mean = preds.mean.squeeze()
          var = preds.variance.squeeze()
          lcb = mean - 2*torch.sqrt(var)
          ucb = mean + 2*torch.sqrt(var)
          plt.figure(figsize=(8,6))
          plt.plot(x_list, mean.detach().numpy())
          plt.fill_between(x_list, lcb.detach().numpy(), ucb.detach().numpy(), alpha=0.2)
          plt.scatter(X.detach().numpy(), Y.detach().numpy(), color ='red', marker ='*')
          plt.xlabel('Input',fontdict=font_axis_publish)
          plt.ylabel('Target',fontdict=font_axis_publish)
      else:
        print("Too many input dimensions to plot!")

    # return the trained model
    return model

def Optimize_acqf(acq, xL, xU, num_restarts=10, raw_samples=1000):
    """
    -----------
    Args:
    acq: the specified acquisition function -- botorch class
    xL: Lower limit of X (input variables) -- torch tensor (1d)
    xU: Upper limit of Y (output variables) -- torch tensor (1d)
    num_restarts: Number of multi-starts to perform -- integer
    raw_samples: Number of random initial samples to use to generate multi-start candidates

    ----------
    returns:

    new_point : next point (x) to sample to obtain f(x) -- torch tensor    
    acq_val : the value of the acquisition function at this point -- torch tensor    
    """  

    # write out bounds in botorch notation
    nx = xL.shape[0]
    bounds = torch.tensor([(xL[j], xU[j]) for j in range(nx)]).T

    # run the optimization routine and extract the required values
    new_point, acq_value_list = optimize_acqf(acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples, options={})
    try:
      acq_val = acq_value_list.numpy()
      
    except:
      print('Optimizing the acqusition failed, so taking single random point')
      new_point = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, nx)
      acq_val = float('nan')

    return new_point, acq_val

