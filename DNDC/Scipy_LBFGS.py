#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L-BFGS Optimization Package through Scipy
@author: donnelly.235
"""

import os
import torch
import sys
from torch.quasirandom import SobolEngine
import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from calib_objective import simulator_run, objective_function
from DNDCrun import DNDC
os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 3
dim = 34
def eval_objective(x):
   
    theta = torch.tensor(x)
    sr1, sr2, sr3, sr4 = simulator_run(x)
    return np.array(objective_function(sr1,sr2,sr3,sr4))

# Initial Point to Begin Optimization
LB = torch.full((1,dim),0.).squeeze(0)
UB = torch.full((1,dim),1.).squeeze(0)
bounds = torch.stack((LB,UB))

torch.random.manual_seed(seed)
sobol = SobolEngine(dimension=dim, scramble=True)
train_t = torch.empty(1)
train_t = sobol.draw(n=1)
train_t = (train_t*(bounds[1]-bounds[0])+bounds[0]).squeeze().numpy()

Xeval = []
Feval = []
Neval = 0

def func_with_callback(x):
  global Xeval, Feval, Neval
  fx = eval_objective(x)
  Xeval.append(x)
  Feval.append(fx)
  Neval += 1
  
  Feval_best = [min(Feval[0:i]) for i in range(1,len(Feval)+1)]
  print("Iteration: %03d ------- Current Best Loss: %3.3f" %(Neval,Feval_best[0].item()))
  return fx

bounds = np.repeat(np.array([[0,1],[0,1]]),dim//2,axis=0)
res = minimize(func_with_callback, train_t, method = 'L-BFGS-B', bounds=bounds,tol=1e-7) #options={'eps':0.00001})

Feval_best = [min(Feval[0:i]) for i in range(1,len(Feval)+1)]
   
font_axis_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 22,
        }
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16

#best = 0 # best solution found over 100 rounds of optimization
fig = plt.figure(figsize=(10,8))
plt.plot(Feval_best, c='g', label='Scipy-LBFGS', lw=3)
plt.xlabel('Evaluations',fontdict=font_axis_publish)
plt.ylabel('Best Loss Value',fontdict=font_axis_publish)
plt.subplots_adjust(left=.15, bottom=.2)
plt.legend()
plt.grid()
plt.yscale("log")
plt.show()

df_LBFGS = Feval_best
df_LBFGS = pd.DataFrame(df_LBFGS)
df_LBFGS.to_csv('df_LBFGS.csv', sep=',', index = False, encoding='utf-8')