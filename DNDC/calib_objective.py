#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Objective for Environmental Model Calibration
@author: donnelly.235
"""
import sys
import torch
from torch import Tensor
from DNDCrun import DNDC

import matplotlib.pyplot as plt

# torch.set_default_dtype(torch.double)

seed = 0   # Use this so that seed is consistent across algos.
simulator = DNDC()
input_dim = simulator.theta_dim
    
def simulator_run(theta):  
    
    # Rescaling:
    LB = simulator.LB
    UB = simulator.UB
    
    theta_scaled = LB + (UB - LB)*theta
    
    # Running model to obtain desired outputs:    
    nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr = simulator.model_run(theta_scaled)
    
    # Obtaining ground truth data from simulator:
    nitrate_tr = simulator.nitrate_tr
    water_tr = simulator.water_tr
    corn_tr = simulator.corn_tr
    wheat_tr = simulator.wheat_tr
    
    nitrate_tr_std = torch.std(nitrate_tr)
    water_tr_std = torch.std(water_tr)
    
    # Calculating sum of weighted squared residuals:
    sr1 = (1/torch.square(nitrate_tr_std))*torch.sum(torch.square(nitrate_tr - nitrate_est_tr))
    sr2 = (1/torch.square(water_tr_std))*torch.sum(torch.square(water_tr - water_est_tr))
    sr3 = torch.sum(torch.square(corn_tr - corn_est_tr))
    sr4 = torch.sum(torch.square(wheat_tr - wheat_est_tr))
    
    output = torch.tensor([sr1,sr2,sr3,sr4])
       
    return output


def objective_function(sr1, sr2, sr3, sr4):
    """
    Defines MLE objective function to optimize, done through minimization of sum of squared residuals.
    """
       
    ssr = sr1 + sr2 + sr3 + sr4
  
    return ssr
             

  

                  

