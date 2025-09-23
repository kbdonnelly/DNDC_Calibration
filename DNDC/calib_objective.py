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
from DNDCrun import DNDC
from turbo_1 import Turbo1


import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.double)

seed = 0   # Use this so that seed is consistent across algos.


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
    run_type = ['TuRBO-1'] # Types accepted: ['Rand','Input','TuRBO-1','TuRBO-C','SCE-UA','L-BFGS-B']
    plotting = False # Option for turning plotting on/off
    seed = 0    
    
    if run_type == ['Rand']:
        theta = torch.rand(1,dim)
        output = f(theta.squeeze(0))
       
    if run_type == ['Input']:
        theta = torch.rand(1,dim)
         
    if run_type == ['TuRBO-1']:
            
        turbo1 = Turbo1(
             f = f,  # Handle to objective function
             lb = np.zeros(f.theta_dim),  # Numpy array specifying lower bounds
             ub = np.ones(f.theta_dim),  # Numpy array specifying upper bounds
             n_init = 2*dim,  # Number of initial bounds from an Latin hypercube design
             max_evals = 200,  # Maximum number of evaluations
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
        df_theta_TuRBO1.to_csv('df_theta_TuRBO1_seed0.csv', sep=',', index = False, encoding='utf-8')
           
        df_output_TuRBO1 =  pd.DataFrame(fX)
        df_output_TuRBO1.to_csv('df_output_TuRBO1_seed0.csv', sep=',', index = False, encoding='utf-8')
         
     # if run_type == ['L-BFGS-B']:


  

                  

