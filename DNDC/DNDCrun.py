#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNDC Model Run
@author: donnelly.235
"""
import pandas as pd 
import torch
from torch import Tensor
import numpy as np
import os
import shutil
import time

class DNDC:
    def __init__(self):
        """
        Defines parameter values and ranges from "DNDCparameters.csv" file.
        -----------------------------------------------------------------------
        params: List of strings of parameters to be located in .dnd file.
        CropID: Tensor of Crop_ID that may be associated with a paramerer.
            -> if this is 0, then the parameter for that row is not crop-specific.
        LB: Tensor of lower bounds for parameters.
        UB: Tensor of upper bounds for parameters.
        nom_params: Tensor of nominal parameter setting. This corresponds to the
            "DNDCNominal.dnd" file that subsequent iterations write over in the 
            "iteration.dnd" file.
             
        """
        # Import parameters, bounds, and nominal values:    
        df1 = pd.read_csv('DNDCparameters.csv')
         
        
        self.params = df1.iloc[:,0].tolist()
        self.CropID = torch.tensor(df1.iloc[:,1].tolist())
        self.LB = torch.tensor(df1.iloc[:,3].tolist())        
        self.UB = torch.tensor(df1.iloc[:,4].tolist())
        self.nom_params = torch.tensor(df1.iloc[:,2].tolist())
                                      
        self.theta_dim = len(df1)
        
        # Import ground truth data for calibration training and testing:
        df2 = pd.read_csv('traintestdata_leaching.csv')
        
        self.nitrate_tr = torch.tensor(df2.iloc[:,0].tolist())
        self.water_tr = torch.tensor(df2.iloc[:,1].tolist())
        self.nitrate_ts = torch.tensor(df2.iloc[:,2].tolist())
        self.water_ts = torch.tensor(df2.iloc[:,3].tolist())
        
        self.wheat_tr = torch.tensor([2206])
        self.corn_tr = torch.tensor([5783])
        self.wheat_ts = torch.tensor([1883])
        self.corn_ts = torch.tensor([4304])
        
            
    def model_run(self, theta):
        """
        Takes new parameter settings, overwrites DNDC input file, and runs
        DNDC model.
        
        Parses for outputs of interest to be used in MLE calculation.

        """
        # Rounding all parameters to correct number of sigfigs and coverts to string for .dnd file:
        
        nom_str = self.nom_params.squeeze(0).tolist()    
        theta_str = theta.squeeze(0).tolist()
        dnd_oldline = [None]*len(theta)
        dnd_newline = [None]*len(theta)
        for i in range(len(theta)):
            theta_str[i] = f'{theta_str[i]:.4f}'
            nom_str[i] = f'{nom_str[i]:.4f}'
            dnd_oldline[i] = self.params[i].ljust(71 - len(nom_str[i])) + nom_str[i]
            dnd_newline[i] = self.params[i].ljust(71 - len(theta_str[i])) + theta_str[i]
        
        # Create input file with new parameters
        filename = 'iteration'
        DefaultPath = "C:\\Input\\DNDCNominal.dnd"
        InputPath = "C:\\Input\\" + filename + ".dnd"

        shutil.copy(DefaultPath, InputPath)
        with open(InputPath, 'r') as file:  # Read in the file
            filedata = file.read()        
        
        # Loop for replacing values in iteration.dnd. This also works for crop-specific data.
        
        for i in range(34):    
            filedata = filedata.replace(dnd_oldline[i], dnd_newline[i])
        
        with open(InputPath, 'w') as file:  # Write the file out again
            file.write(filedata)
        file.close()
            
        start = time.time()
        print('Running DNDC...')
        os.system("cd C:\DNDC")
        os.system("start/wait DNDC95 -s C:/DNDC/batchOut.txt")
        end = time.time()
        print('DNDC run complete in' + ' ' + f'{end-start:.4f}' + ' ' + 'seconds.')
        
        # Obtaining outputs of interest for calibration:
            
        df_nitrate = pd.read_csv('C:\DNDC\Result\Record\Batch\Case1-M\Day_SoilN_1.csv',usecols=[39],header=5)
        df_nitrate = torch.tensor(df_nitrate.to_numpy())    
        
        nitrate_est_tr = df_nitrate[2188:4014,...].squeeze(1)
        
        df_water = pd.read_csv('C:\DNDC\Result\Record\Batch\Case1-M\Day_SoilWater_1.csv',usecols=[15],header=2)
        df_water = torch.tensor(df_water.to_numpy())    
        
        water_est_tr = df_water[2189:4015,...].squeeze(1)
        
        df_crops = pd.read_csv('C:\DNDC\Result\Record\Batch\Case1-M\Multi_year_summary.csv',usecols=[2,3,4],header=2)
        df_crops = torch.tensor(df_crops.to_numpy())    
        
        corn_est_tr = torch.sum(df_crops[5])
        wheat_est_tr = df_crops[6,0]
        
        
       
        return nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr

if __name__ == '__main__':
    
    a = DNDC()
    
    random_test = False # An option if a random test of parameters is desired
    
    if random_test == True:
        LB = a.LB
        UB = a.UB
        rand_init = LB + (UB - LB)*torch.rand(34)
    
        nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr = a.model_run(rand_init)
    
    else: 
        print('Please specify your input theta set.')
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    font_axis_publish = {
            'color':  'black',
            'weight': 'bold',
            'size': 22,
            }
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(nitrate_est_tr.numpy().flatten(), c='r', lw=3)
    plt.plot(a.nitrate_tr.numpy().flatten(), c='g', lw=3)
    plt.xlabel('Day (2014-2018)',fontdict=font_axis_publish)
    plt.ylabel('Nitrate Leaching (kg/ha)',fontdict=font_axis_publish)
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(water_est_tr.numpy().flatten(), c='b', lw=3)
    plt.xlabel('Day (2014-2018)',fontdict=font_axis_publish)
    plt.ylabel('Water Leaching (mm)',fontdict=font_axis_publish)