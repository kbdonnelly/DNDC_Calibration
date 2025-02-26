#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla BO loop
@author: donnelly.235
"""

import numpy as np
import pandas as pd

import gpytorch
import botorch
import torch
import sys

torch.set_default_dtype(torch.float64)
from botorch.utils.sampling import draw_sobol_samples
from torch.quasirandom import SobolEngine
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
from BO_helperfunctions import GP_training, Optimize_acqf

from calib_objective import simulator_run, objective_function

import time
import shutil

# turn off warnings for clean output
import warnings
warnings.filterwarnings('ignore')
     
def BO_optimization_loop(t_train,y_train,seed):
    """
    Vanilla BO loop used for optimization. See "BO_helperfunctions" for related functions.
    
    """      
    
    # Specifying parameters:
    nreps = 1         # number of full optimiztion rounds
    total_evals = 50  # num total budget (includes n_init)
    n_init = 34       # num init samples
    n_sa = 1          # num sample avg
    dim = 34
    
    # Data for BO method:
    BO_losses = torch.zeros(nreps, total_evals-n_init)
    Best_loss = torch.zeros(nreps, total_evals-n_init-1)
    BO_vec = torch.zeros(nreps, total_evals-n_init, n_init)
    BO_y = torch.zeros(nreps, total_evals-n_init)
    
    # Fix seed
    np.random.seed(seed)
    rand_ic = torch.tensor(np.random.uniform(np.exp(-3),np.exp(3), (1,n_sa)))
    
    for j in range(nreps):
    
        # bounds
        LB = torch.full((1,dim),0.).squeeze(0)
        UB = torch.full((1,dim),1.).squeeze(0)
        bounds = torch.stack((LB,UB))
         
        t_init = t_train.squeeze(-1)
        y_init = y_train.squeeze(-1)
        
        # Initial data 
        torch.random.manual_seed(seed)
        sobol = SobolEngine(dimension=dim, scramble=True)
        train_t = torch.empty(total_evals, dim)
        train_t[:n_init,:] = sobol.draw(n=n_init)
        #train_t[:n_init,:] = train_t[:n_init,:]*(bounds[1]-bounds[0])+bounds[0]
          
        # Evaluate function
        train_y = torch.empty(total_evals, 4)
        for i in range(n_init):
            train_y[i,:] = simulator_run(train_t[i,:])
            BO_losses = objective_function(train_y[i,:])
        #train_y[:n_init,:] = simulator_run(train_t[:n_init,:])
            
        #BO_losses = objective_function(train_y[:n_init,:])
        
        for k in range(n_init,total_evals-1):
            #fit model
            model = GP_training(train_t[:k], BO_losses[:k], 'RBF', noise_free=True)
        
            #build and opt EI Acq fun
            best_value = BO_losses.min().item()
            ei = ExpectedImprovement(model=model, best_f=best_value, maximize=False)
            candidates, acq_val = optimize_acqf(ei, bounds=bounds,  q=1, num_restarts=10, raw_samples=1000)
            
            # observe new values 
            new_t = candidates.detach()
            train_t[k,:] = new_t
            yi = 0
            for i in range(n_sa):
                yi += simulator_run(train_t[k,:].unsqueeze(0))
                train_y[k+1,:] = yi/n_sa
                
            # save results
            BO_losses = objective_function(train_y[:k+1,:])
            Best_loss[:,k-n_init] = BO_losses.min()
            # BO_vec[j,k+1, :] = new_x
            # BO_y[j,k+1] = train_y[k+n_init,:]
                  
            
            # print current results
            print("Round %03d --- it: %03d, best: %3.3f" % 
                  (j+1, 
                    k+1, 
                  BO_losses.min().item() ))
    return Best_loss