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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        self.nitrate_ts = torch.tensor(df2.iloc[:,2].tolist())[0:730]
        self.water_ts = torch.tensor(df2.iloc[:,3].tolist())[0:730]
        
        self.wheat_tr = torch.tensor([2206])
        self.corn_tr = torch.tensor([5783])
        self.wheat_ts = torch.tensor([1883])
        self.corn_ts = torch.tensor([4304])
        
        # Specify weights for the output sensors:
            
        self.weights = torch.tensor([1.0,1.0,1.0,1.0])
        
            
    def __call__(self, theta):
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
        os.system("cd C:\\DNDC")
        os.system("start/wait DNDC95 -s C:/DNDC/batchOut.txt")
        end = time.time()
        print('DNDC run complete in' + ' ' + f'{end-start:.4f}' + ' ' + 'seconds.')
        
        # Obtaining outputs of interest for calibration:
            
        df_nitrate = pd.read_csv('C:\\DNDC\\Result\\Record\\Batch\\Case1-M\\Day_SoilN_1.csv',usecols=[39],header=5)
        df_nitrate = torch.tensor(df_nitrate.to_numpy())    
        
        nitrate_est_tr = df_nitrate[2188:4013,...].squeeze(1)
        self.nitrate_est_ts = df_nitrate[4013:4743,...].squeeze(1)
                                                           
        df_water = pd.read_csv('C:\\DNDC\\Result\\Record\\Batch\\Case1-M\\Day_SoilWater_1.csv',usecols=[15],header=2)
        df_water = torch.tensor(df_water.to_numpy())    
        
        water_est_tr = df_water[2189:4014,...].squeeze(1)
        self.water_est_ts = df_water[4014:4745,...].squeeze(1)
        
        df_crops = pd.read_csv('C:\\DNDC\\Result\\Record\\Batch\\Case1-M\\Multi_year_summary.csv',usecols=[2,3,4],header=2)
        df_crops = torch.tensor(df_crops.to_numpy())    
        
        corn_est_tr = (df_crops[5,1] + df_crops[5,2])*(0.8) + df_crops[5,0]
        wheat_est_tr = df_crops[6,0]
        self.corn_est_ts = (df_crops[8,1] + df_crops[8,2])*(0.8) + df_crops[8,0]
        self.wheat_est_ts = df_crops[9,0]
        
        df_crop1 = pd.read_csv('C:\\DNDC\\Result\\Record\\Batch\\Case1-M\\Day_FieldCrop_1.csv',usecols=[35,36,38],header=4)
        df_crop1 = torch.tensor(df_crop1.to_numpy())[1824:3649,...]
        df_crop1_leaf = df_crop1[:,0]
        df_crop1_stem = df_crop1[:,1]
        df_crop1_grain = df_crop1[:,2]
        self.total_crop1_biomass = df_crop1_grain
        self.total_crop1_biomass[149:264,...] = df_crop1_grain[149:264,...]+(0.8)*(df_crop1_leaf[149:264,...] + df_crop1_stem[149:264,...])
        self.total_crop1_biomass[497:552,...] = df_crop1_grain[497:552,...]
        self.total_crop1_biomass[1243:1359,...] = df_crop1_grain[1243:1359,...]+(0.8)*(df_crop1_leaf[1243:1359,...] + df_crop1_stem[1243:1359,...])
        self.total_crop1_biomass[1566:1647,...] =  df_crop1_grain[1566:1647,...]
        self.total_crop1_biomass[767:942,...]= torch.zeros(175)
        
        return nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr

if __name__ == '__main__':
    
    a = DNDC()
    dim = a.theta_dim
    run_type = ['Input'] # Types accepted: ['Rand','Input','Nominal']
    
    plotting = True # Option for turning plotting on/off
    
    if run_type == ['Rand']:
        LB = a.LB
        UB = a.UB
        theta = torch.rand(dim)

        # Rescaling:    
        theta = LB + (UB - LB)*theta
        
        nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr = a(theta)
        
    
    if run_type == ['Input']:
        LB = a.LB
        UB = a.UB
        # df1_param = pd.read_csv('df_X_SCEUA1.csv')
        # theta = torch.tensor(df1_param.to_numpy())
        # theta = LB + (UB - LB)*theta
        # df1_obj = pd.read_csv('df_Y_SCEUA1.csv')
        # obj = torch.tensor(df1_obj.to_numpy())
        # theta_best = theta[torch.argmin(obj)]
        
        # theta = a.nom_params
        
        theta = torch.tensor([0.524, 0.384, 0.288, 0.722, 0.713, 0.202, 0.718, 0.07,  0.97,  0.595, 0.804, 0.336,
                              0.236, 0.825, 0.92,  0.186, 0.128, 0.488, 0.108, 0.823, 0.551, 0.802, 0.494, 0.581,
                              0.738, 0.517, 0.49,  0.196, 0.344, 0.306, 0.625, 0.71,  0.591, 0.456])
        theta = LB + (UB - LB)*theta
        
        nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr = a(theta)
        nitrate_full = torch.cat([nitrate_est_tr,a.nitrate_est_ts], dim=0)
        water_full = torch.cat([water_est_tr,a.water_est_ts], dim=0)
                
        # df_leaching_sensors =  pd.DataFrame([nitrate_full.numpy(), water_full.numpy()]).T
        # df_leaching_sensors.to_csv('df_leaching_bhattarai_092225.csv', sep=',', index = False, encoding='utf-8')
        # df_biomass_sensors = pd.DataFrame([a.total_crop1_biomass.numpy()]).T
        # df_biomass_sensors.to_csv('df_biomass_bhattarai_092225.csv', sep=',', index = False, encoding='utf-8')
        
    if run_type == ['Nominal']:
        theta = a.nom_params
        
        nitrate_est_tr, water_est_tr, corn_est_tr, wheat_est_tr = a(theta)
        
    
    else: 
               
        LB = a.LB
        UB = a.UB
        
 
    if plotting == True:  

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import datetime
        
        
        #######################################################################
        # First, we define our date ranges used for calibration and validation.
        #######################################################################
        
        start_tr_date = datetime.date(2014,1,1)
        end_tr_date = datetime.date(2018,12,31)
        start_ts_date = datetime.date(2019,1,1)
        end_ts_date = datetime.date(2020,12,31)
        tr_dates = [start_tr_date + datetime.timedelta(days=i) for i in range((end_tr_date - start_tr_date).days)]
        ts_dates = [start_ts_date + datetime.timedelta(days=i) for i in range((end_ts_date - start_ts_date).days)]
        all_dates = [start_tr_date + datetime.timedelta(days=i) for i in range((end_ts_date - start_tr_date).days)]
        
        biomass_start_date = datetime.date(2013,1,1)
        biomass_end_date = datetime.date(2017,12,31)
        biomass_all_dates = [biomass_start_date + datetime.timedelta(days=i) for i in range((biomass_end_date - biomass_start_date).days)]
        
        # Here, we also calculate monthly averages for sensors of interest:
        num_months_tr = (end_tr_date.year - start_tr_date.year) * 12 + end_tr_date.month - start_tr_date.month
        num_months_ts = (end_ts_date.year - start_ts_date.year) * 12 + end_ts_date.month - start_ts_date.month

        # Adjust tensor size to match the number of months:
        monthly_nitrate_tr = torch.empty(4, num_months_tr)  # Correct size based on the number of months
        monthly_water_tr = torch.empty(4, num_months_tr)
        monthly_nitrate_ts = torch.empty(4, num_months_ts)
        monthly_water_ts = torch.empty(4, num_months_ts)

        def monthly_leaching(daily_leaching_data):
            # dates = [start_tr_date + datetime.timedelta(days=i) for i in range((end_ts_date - start_tr_date).days)]
            dates = pd.date_range(start=start_tr_date, periods=2557, freq="D")
            dates = dates[~((dates.month == 2) & (dates.day == 29))].to_series().iloc[:2557].index
            df = pd.DataFrame({
                'dates': dates,
                'leaching': daily_leaching_data.numpy()
            })
            df['dates'] = pd.to_datetime(df['dates'])
            df['month'] = df['dates'].dt.to_period('M')  # Group by month
            monthly_leaching_data = df.groupby('month')['leaching'].mean()
            return monthly_leaching_data
        
        # Importing Bhattari et al. (2022) Results:
        
        df_leaching = pd.read_csv('df_leaching_bhattarai_092225.csv')
        df_biomass = pd.read_csv('df_biomass_bhattarai_092225.csv')
         
        nitrate_bhattarai =  torch.tensor(df_leaching.iloc[:,0].tolist())
        water_bhattarai =  torch.tensor(df_leaching.iloc[:,1].tolist())
        biomass_bhattarai = torch.tensor(df_biomass.iloc[:,0].tolist())
        
        # Calculating monthly leaching averages:
        
        nitrate_est_monthly = monthly_leaching(torch.cat([nitrate_est_tr,a.nitrate_est_ts], dim=0))
        water_est_monthly = monthly_leaching(torch.cat([water_est_tr,a.water_est_ts], dim=0))
        nitrate_gt_monthly = monthly_leaching(torch.cat([a.nitrate_tr,a.nitrate_ts],dim=0))
        water_gt_monthly = monthly_leaching(torch.cat([a.water_tr,a.water_ts],dim=0))
        nitrate_bhattarai_monthly = monthly_leaching(nitrate_bhattarai)
        water_bhattarai_monthly = monthly_leaching(water_bhattarai)
        
        #######################################################################
        # Time series calibration plots
        #######################################################################
        
        common_start_date = biomass_all_dates[0]
        common_end_date = datetime.date(2020,12,31)
        monthly_dates = pd.date_range(start=common_start_date, end=common_end_date, freq="MS").date
        
        
        fig, ax = plt.subplots(3, 1, figsize=(24, 18))
        
        # Monthly Water Leaching Plot:
            
        ax[0].step(monthly_dates[12:97], np.array(water_est_monthly), c='orange', lw=3)
        ax[0].step(monthly_dates[12:97], np.array(water_bhattarai_monthly), c='b', lw=3,linestyle=":")
        ax[0].plot(monthly_dates[12:97], np.array(water_gt_monthly), c='k', lw=3, linestyle="none",marker="x",markersize=12)
        ax[0].axvspan(start_ts_date, common_end_date,color='gray',alpha=0.3)
        ax[0].legend(['TuRBO-1','Bhattarai et al. (2022)','Ground Truth', 'Validation Region'], loc='upper center', fontsize=16, ncol=4)
        ax[0].set_ylabel('Water Leaching (mm)', fontsize=20)
        ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        ax[0].set_xlim(common_start_date, common_end_date)
        ax[0].tick_params(axis='x', which='major', pad=15,labelsize=16)
        ax[0].tick_params(axis='y',labelsize=16)
        ax[0].spines['bottom'].set_linewidth(2)
        ax[0].spines['left'].set_linewidth(2)
        ax[0].set_ylim([0, 4])
        ax[0].grid(True)
        
        # Monthly Nitrate Leaching Plot:
        
        ax[1].step(monthly_dates[12:97], np.array(nitrate_est_monthly), c='orange', lw=3)
        ax[1].step(monthly_dates[12:97], np.array(nitrate_bhattarai_monthly), c='b', lw=3,linestyle=":")
        ax[1].plot(monthly_dates[12:97], np.array(nitrate_gt_monthly), c='k', lw=3, linestyle="none",marker="x",markersize=12)
        ax[1].axvspan(start_ts_date, common_end_date,color='gray',alpha=0.3)
        ax[1].legend(['TuRBO-1','Bhattarai et al. (2022)','Ground Truth', 'Validation Region'], loc='upper center', fontsize=16, ncol=4)
        ax[1].set_ylabel('Nitrate Leaching (kg N/ha)', fontsize=20)
        ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        ax[1].set_xlim(common_start_date, common_end_date)
        ax[1].tick_params(axis='x', which='major', pad=15,labelsize=16)
        ax[1].tick_params(axis='y',labelsize=16)
        ax[1].spines['bottom'].set_linewidth(2)
        ax[1].spines['left'].set_linewidth(2)
        ax[1].set_ylim([0, 2.5])
        ax[1].grid(True)
        
        # Crop Biomass Yield Plot:        

        ax[2].plot(biomass_all_dates, a.total_crop1_biomass.numpy().flatten(), c='orange', lw=3)
        ax[2].plot(biomass_all_dates, biomass_bhattarai.numpy().flatten(), c='blue', lw=3,linestyle=":")
        
        ax[2].plot(biomass_all_dates[264], a.corn_tr, c='k', linestyle="none", marker="x", markersize=12)
        ax[2].annotate('Corn Silage Yield (2013)', (biomass_all_dates[264], a.corn_tr), textcoords="offset points", xytext=(10,10), ha='left', fontsize=16,fontweight='bold')
        
        ax[2].plot(biomass_all_dates[552], a.wheat_tr, c='k', linestyle="none", marker="x", markersize=12)
        ax[2].annotate('Wheat Grain Yield (2014)', (biomass_all_dates[552], a.wheat_tr), textcoords="offset points", xytext=(10,10), ha='left', fontsize=16,fontweight='bold')
        
        ax[2].plot(biomass_all_dates[1359], a.corn_ts, c='k', linestyle="none", marker="x", markersize=12)
        ax[2].annotate('Corn Silage Yield (2016)', (biomass_all_dates[1359], a.corn_ts), textcoords="offset points", xytext=(10,10), ha='left', fontsize=16,fontweight='bold')
        
        ax[2].plot(biomass_all_dates[1647], a.wheat_ts, c='k', linestyle="none", marker="x", markersize=12)
        ax[2].annotate('Wheat Grain Yield (2017)', (biomass_all_dates[1647], a.wheat_ts), textcoords="offset points", xytext=(10,10), ha='left', fontsize=16,fontweight='bold')
        
        ax[2].legend(['TuRBO-1','Bhattaria et al. (2022)','Ground Truth'], loc='upper center', fontsize=16, ncol=3)
        ax[2].set_ylabel('Crop Biomass (kg/ha)', fontsize=20)
        ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        ax[2].set_xlim(common_start_date, common_end_date)  # <-- Updated here
        ax[2].set_ylim([0,7000])
        ax[2].tick_params(axis='x', which='major', pad=15,labelsize=16)
        ax[2].tick_params(axis='y',labelsize=16)
        ax[2].spines['bottom'].set_linewidth(2)
        ax[2].spines['left'].set_linewidth(2)
        ax[2].grid(True)
    
        plt.tight_layout()
        plt.show()
        
        #######################################################################
        # Cumulative Leaching Plots
        #######################################################################

        fig, ax = plt.subplots(2, 1, figsize=(12, 12))
        ax[0].plot(all_dates[0:2555],np.cumsum(torch.cat([nitrate_est_tr,a.nitrate_est_ts],dim=0).numpy()), color='orange',lw=3,linestyle="--")
        ax[0].plot(all_dates[0:2555],np.cumsum(nitrate_bhattarai.numpy()),color='purple',lw=3,linestyle="--")
        ax[0].plot(all_dates[0:2555],np.cumsum(torch.cat([a.nitrate_tr,a.nitrate_ts],dim=0).numpy()), color='g',lw=3,linestyle=":")
        ax[0].axvspan(start_ts_date, common_end_date,color='gray',alpha=0.3)
        ax[0].legend(['TuRBO-1','Bhattarai et al. (2022)','Ground Truth','Validation Region'],fontsize=12)
        ax[0].set_ylabel('Cumulative Nitrate Leaching (kg N/ha)',fontsize=16)
        ax[0].set_xlabel('Years',fontsize=16)
        ax[0].tick_params(axis='x',labelsize=12)
        ax[0].tick_params(axis='y',labelsize=12)
        ax[0].grid(True)


        ax[1].plot(all_dates[0:2555],np.cumsum(torch.cat([water_est_tr,a.water_est_ts],dim=0).numpy()), color='orange',lw=3,linestyle="--")
        ax[1].plot(all_dates[0:2555],np.cumsum(water_bhattarai.numpy()),color='purple',lw=3,linestyle="--")
        ax[1].plot(all_dates[0:2555],np.cumsum(torch.cat([a.water_tr,a.water_ts],dim=0).numpy()), color='g',lw=3,linestyle=":")
        ax[1].axvspan(start_ts_date, common_end_date,color='gray',alpha=0.3)
        ax[1].legend(['TuRBO-1','Bhattarai et al. (2022)','Ground Truth','Validation Region'],fontsize=12)
        ax[1].set_ylabel('Cumulative water Leaching (mm)',fontsize=16)
        ax[1].set_xlabel('Years',fontsize=16)
        ax[1].tick_params(axis='x',labelsize=12)
        ax[1].tick_params(axis='y',labelsize=12)
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()
        