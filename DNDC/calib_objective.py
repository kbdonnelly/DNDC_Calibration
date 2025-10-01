#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Objective for Environmental Model Calibration
@author: donnelly.235
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.quasirandom import SobolEngine
import sceua

from DNDCrun import DNDC
from turbo_1 import Turbo1
from turbo_c import cTurbo1
SMOKE_TEST = False

import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.dates as mdates
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# torch.set_default_dtype(torch.double)

class ObjFunc:
    def __init__(self):  
        self.simulator = DNDC()
        self.theta_dim = self.simulator.theta_dim
        self.nom_params = self.simulator.nom_params
        
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
    run_type = ['SCE-UA'] # Types accepted: ['Rand','Input','TuRBO-1','TuRBO-C','SCE-UA','L-BFGS-B']
    plotting_allseeds = False # Option for turning plotting on/off
    plotting_bestseed = False
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
        bounds = [(0, 1)] * f.theta_dim
        
        
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
        
        # Output results
        print("\nOptimization Complete:")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Optimal x: {result.x}")
        print(f"Optimal f(x): {result.fun}")
        
        if len(F_eval) < 2000:
          for i in range(2000-len(F_eval)):
            F_eval.append(F_eval[-1])
    
    if run_type == ['SCE-UA']:
        # Store evaluations:
        X_eval = []
        F_eval = []
        bounds = [(0, 1)] * f.theta_dim
        
        # Define objective function:
        def objective(x):
            x_tensor = torch.tensor(x)
            fx = torch.sum(f(x_tensor)).numpy()
            X_eval.append(x.copy())
            F_eval.append(fx)
            print(f"Eval {len(F_eval):03d}: = {fx:.6f}")
            return fx                 
        
        # Run optimization
        result = sceua.minimize(
            objective,
            bounds,
            args=(),
            n_complexes=None,
            n_points_complex=None,
            alpha=1.0,
            beta=0.5,
            max_evals=2000,
            max_iter=1000,
            max_tolerant_iter=1000,
            tolerance=1e-06,
            x_tolerance=1e-08,
            seed=seed,
            pca_freq=1,
            pca_tol=0.001,
            x0=None,
            max_workers=1
            )
            
        # Output results
        print("\nOptimization Complete:")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Optimal x: {result.x}")
        print(f"Optimal f(x): {result.fun}")
        
        df_theta_SCEUA =  pd.DataFrame(result.xv)
        df_theta_SCEUA.to_csv('df_theta_SCEUA_seed0.csv', sep=',', index = False, encoding='utf-8')
           
        df_output_SCEUA =  pd.DataFrame(result.funv)
        df_output_SCEUA.to_csv('df_output_SCEUA_seed0.csv', sep=',', index = False, encoding='utf-8')
        
        
        
    if plotting_allseeds == True:  
        
        # Select which methods are to be plotted:
        
        methods = ['TuRBO-1','TuRBO-C']
        colors = ['orange','red']
        
        # Imports data from methods selected, and plots:
        
        nom_params_scaled = (f.nom_params-f.LB)/(f.UB-f.LB)
        baseline = torch.sum(f(nom_params_scaled))
        fig, ax = plt.subplots(figsize=(8, 6))           
        for i in range(len(methods)):
            folder_path = 'C:\\DNDC\\Optimization_Data\\' + methods[i] + '\\ObjFunc'
            all_files = os.listdir(folder_path)
            df = pd.DataFrame() 
            for j in range(len(all_files)):
                df_data = pd.read_csv(folder_path + '\\' + all_files[j])
                df.loc[:,j] = pd.DataFrame(np.minimum.accumulate(df_data.to_numpy()))
            # Calculate average loss, create bounds:
            average = torch.mean(torch.tensor(df.to_numpy())[0:2000],dim=1)
            std = torch.std(torch.tensor(df.to_numpy())[0:2000],dim=1)
            lcb = average - std
            ucb = average + std
            x = np.arange(len(average))
            print(average[-1])
            plt.plot(average, marker="s", lw=3,c=colors[i],markevery=100,label=methods[i])
            plt.fill_between(x, lcb.detach().numpy(), ucb.detach().numpy(), color=colors[i], alpha=0.2,label='_nolegend_')
        plt.axhline(baseline.item(),xmin=0,xmax=2000,lw=3,linestyle="--",color='k',label = 'Bhattarai et al. (2022)')  
        plt.legend(loc='upper right',fontsize=12)    
        # ax.set_yscale('log')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel("NRMSE", fontsize = 16)
        plt.xlabel("Model Evaluations", fontsize = 16)
        plt.ylim([0.09,1.1])
        plt.xlim([0,500])
        plt.grid(True)
        plt.tight_layout()
        plt.show()    
     
    if plotting_bestseed == True:  

        import matplotlib.pyplot as plt
        import scipy.stats as stats
        import matplotlib.dates as mdates
        import datetime
        
        # Select which methods are to be plotted:
            
        methods = ['TuRBO-C']
        colors = ['red']
        
        # Imports data from methods selected, and determines lowest NRMSE:
            
        
        nom_params_scaled = (f.nom_params-f.LB)/(f.UB-f.LB)
        baseline = torch.sum(f(nom_params_scaled))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(methods)):
            folder_path = 'C:\\DNDC\\Optimization_Data\\' + methods[i] + '\\ObjFunc'
            all_files = os.listdir(folder_path)
            df = pd.DataFrame() 
            for j in range(len(all_files)):
                df_data = pd.read_csv(folder_path + '\\' + all_files[j])
                df.loc[:,j] = pd.DataFrame(np.minimum.accumulate(df_data.to_numpy()))
            allseeds = torch.tensor(df.to_numpy())[0:2000]    
            best_nrmse_index = torch.argmin(allseeds[-1])
            best_csv_path = folder_path + '\\' + 'df_output_' + methods[i] + '_seed' + str(best_nrmse_index.item()) +'.csv'
            plt.plot(np.minimum.accumulate(allseeds[:,best_nrmse_index].numpy()), marker="s", lw=3,c=colors[i],markevery=100,label=methods[i])
            plt.plot(pd.read_csv(best_csv_path).iloc[:,0].to_numpy(), marker=".",c=colors[i],linestyle="none",label=methods[i]+': Iterations',alpha=0.2)
        
        plt.axhline(baseline.item(),xmin=0,xmax=2000,lw=3,linestyle="--",color='k',label = 'Bhattarai et al. (2022)')  
        plt.legend(loc='upper right',fontsize=12)    
        # ax.set_yscale('log')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel("NRMSE", fontsize = 16)
        plt.xlabel("Model Evaluations", fontsize = 16)
        plt.ylim([0.09,1.1])
        plt.xlim([0,2000])
        plt.grid(True)
        plt.tight_layout()
        plt.show()    
             
        
        