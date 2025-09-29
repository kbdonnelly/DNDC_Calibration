#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Objective for Environmental Model Calibration
@author: donnelly.235
"""
import sys
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.quasirandom import SobolEngine
from scipy.optimize import minimize

from DNDCrun import DNDC
from turbo_1 import Turbo1
from turbo_c import cTurbo1
SMOKE_TEST = False

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# torch.set_default_dtype(torch.double)

class ObjFunc:
    def __init__(self):  
        self.simulator = DNDC()
        self.theta_dim = self.simulator.theta_dim
        
        self.LB = self.simulator.LB
        self.UB = self.simulator.UB
        
    def __call__(self,theta):
        
        theta_scaled = self.LB + (self.UB - self.LB)*theta
        
        # Running model to obtain desired outputs:    
        nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr = self.simulator(theta_scaled)
        
        # Obtaining ground truth data from simulator:
        nitrate_tr = self.simulator.nitrate_tr[0:1826,...]
        water_tr = self.simulator.water_tr[0:1826,...]
        corn_tr = self.simulator.corn_tr
        wheat_tr = self.simulator.wheat_tr
           
        # Calculating sum of weighted squared residuals:
        sr1 = torch.sqrt((1/len(nitrate_tr))*torch.sum(torch.square(nitrate_tr - nitrate_est_tr)))/torch.max(nitrate_tr)
        sr2 = torch.sqrt((1/len(water_tr))*torch.sum(torch.square(water_tr - water_est_tr)))/torch.max(water_tr)
        sr3 = torch.sqrt(torch.sum(torch.square(corn_tr - corn_est_tr)))/corn_tr
        sr4 = torch.sqrt(torch.sum(torch.square(wheat_tr - wheat_est_tr)))/wheat_tr
        
        output = torch.tensor([sr1,sr2,sr3,sr4])
           
        return output

if __name__== '__main__':
    f = ObjFunc()
    dim = f.theta_dim
    run_type = ['L-BFGS-B'] # Types accepted: ['Rand','Input','TuRBO-1','TuRBO-C','SCE-UA','L-BFGS-B']
    plotting = False # Option for turning plotting on/off
    seed = 0
       
    if run_type == ['Input']:
        theta = torch.rand(1,dim)
    
    if run_type == ['Rand']:
        theta = torch.rand(1,dim)
        output = f(theta.squeeze(0))    
    
    if run_type == ['TuRBO-1']:
            
        turbo1 = Turbo1(
             f = f,  # Handle to objective function
             lb = np.zeros(f.theta_dim),  # Numpy array specifying lower bounds
             ub = np.ones(f.theta_dim),  # Numpy array specifying upper bounds
             n_init = 2*dim,  # Number of initial bounds from an Latin hypercube design
             max_evals = 70,  # Maximum number of evaluations
             batch_size = 10,  # How large batch size TuRBO uses
             verbose = True,  # Print information from each batch
             use_ard = True,  # Set to true if you want to use ARD for the GP kernel
             max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
             n_training_steps = 50,  # Number of steps of ADAM to learn the hypers
             min_cuda = 1024,  # Run on the CPU for small datasets
             device = "cpu",  # "cpu" or "cuda"
             dtype = "float64",  # float64 or float32
             seed = seed
         )
        turbo1.optimize()
        
        X = turbo1.X  # Evaluated points
        fX = turbo1.fX  # Observed values
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]
        
        print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
         
        df_theta_TuRBO1 =  pd.DataFrame(X)
        df_theta_TuRBO1.to_csv('df_theta_TuRBO1_seed1.csv', sep=',', index = False, encoding='utf-8')
           
        df_output_TuRBO1 =  pd.DataFrame(fX)
        df_output_TuRBO1.to_csv('df_output_TuRBO1_seed1.csv', sep=',', index = False, encoding='utf-8')
         
    if run_type == ['TuRBO-C']:
        # Setting initial parameters:
        batch_size = 10
        n_init = 2 * dim
        max_evals = 2000 if not SMOKE_TEST else 10
        max_cholesky_size = float("inf")  # Always use Cholesky
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
       
        def g(x):
            return -f(x)
        
        cturbo1 = cTurbo1(g, weights, dim, n_init, max_evals, batch_size, max_attempts=5, acqf='ts', max_data_length=500, n_candidates=2000, num_restarts=5, raw_samples=2056, seed=seed)
        
        cturbo1.optimize()
        
        Y_cturbo = cturbo1.fX
        Y_opt_cturbo = Y_cturbo.max().item()
        X_opt_cturbo = cturbo1.X[torch.argmax(cturbo1.fX)].detach().numpy()
        print("Best-found input:", X_opt_cturbo)
        print("Best-found objective value:", Y_opt_cturbo)
        
        df_theta_TuRBOC =  pd.DataFrame(cturbo1.X)
        df_theta_TuRBOC.to_csv('df_theta_TuRBOC_seed4.csv', sep=',', index = False, encoding='utf-8')
           
        df_output_TuRBOC =  pd.DataFrame(Y_cturbo)
        df_output_TuRBOC.to_csv('df_output_TuRBOC_seed4.csv', sep=',', index = False, encoding='utf-8')

    if run_type == ['L-BFGS-B']:
        # Store evaluations:
        X_eval = []
        F_eval = []

        # Define objective function:
        def objective(x):
            x_tensor = torch.tensor(x)
            fx = -torch.sum(f(x_tensor)).numpy()
            X_eval.append(x.copy())
            F_eval.append(fx)
            print(f"Eval {len(F_eval):03d}: = {fx:.6f}")
            return fx          

        # Initial parameters:
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        x0 = sobol.draw(n=1).to(dtype=dtype, device=device)
        x0 = x0.squeeze(0).numpy()
        bounds = [tuple(t) for t in np.tile([0, 1], (f.theta_dim, 1))]
        
        
        # Run optimization
        result = minimize(
            fun=objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxfun': 100,    # Max number of function evaluations
                'disp': True      # Show optimizer messages
            }
        )
        
        # # Output results
        # print("\nOptimization Complete:")
        # print(f"Success: {result.success}")
        # print(f"Message: {result.message}")
        # print(f"Optimal x: {result.x}")
        # print(f"Optimal f(x): {result.fun}")
        
        # if len(F_eval) < 2000:
        #   for i in range(2000-len(F_eval)):
        #     F_eval.append(F_eval[-1])
                
        
        
        