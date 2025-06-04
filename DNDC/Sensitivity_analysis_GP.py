# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:47:27 2025

@author: tang.1856, donnelly.235
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
from calib_objective import simulator_run
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import torch
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

seed = 0 # For replicates
dim = 34 # number of parameters

# Importing parameters for case study of interest:
df_param = pd.read_csv('df_X_cturbo.csv')
param = torch.tensor(df_param.to_numpy())
df_obj = pd.read_csv('df_Y_cturbo.csv')
obj = torch.tensor(df_obj.to_numpy())

best_found = param[torch.argmin(obj)].numpy()
 
# Specify the hypercube where we generate sobol samples:
    
lb_SA = best_found - 0.025 # we can also play around with the size of hypercube
ub_SA = best_found + 0.025

np.clip(lb_SA, 0, 1)
np.clip(ub_SA, 0, 1)

Ninit = 500 # number of data point generated around the best found parameter

train_X = torch.tensor(lb_SA) + torch.tensor(ub_SA - lb_SA)*torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit) # generate training data to train local GP
train_Y = []

for train_x in train_X:
    simulation_output = simulator_run(train_x) # run the simulation
    train_Y.append(simulation_output) 
    
train_Y = torch.stack(train_Y)

# Build model list:
    
model_list = []
for nx in range(4):
    covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
    model_list.append(SingleTaskGP(train_X.to(torch.float64), train_Y[:,nx].unsqueeze(1).to(torch.float64), outcome_transform=Standardize(m=1), covar_module=covar_module))
model = ModelListGP(*model_list)
mll = SumMarginalLogLikelihood(model.likelihood, model)

# Fit the GPs
fit_gpytorch_mll(mll)

problem = {
    'num_vars':dim,
    'names':['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34'],
    'bounds':[[lb_SA[0], ub_SA[0]],
              [lb_SA[1], ub_SA[1]],
              [lb_SA[2], ub_SA[2]],
              [lb_SA[3], ub_SA[3]],
              [lb_SA[4], ub_SA[4]],
              [lb_SA[5], ub_SA[5]],
              [lb_SA[6], ub_SA[6]],
              [lb_SA[7], ub_SA[7]],
              [lb_SA[8], ub_SA[8]],
              [lb_SA[9], ub_SA[9]],
              [lb_SA[10], ub_SA[10]],
              [lb_SA[11], ub_SA[11]],
              [lb_SA[12], ub_SA[12]],
              [lb_SA[13], ub_SA[13]],
              [lb_SA[14], ub_SA[14]],
              [lb_SA[15], ub_SA[15]],
              [lb_SA[16], ub_SA[16]],
              [lb_SA[17], ub_SA[17]],
              [lb_SA[18], ub_SA[18]],
              [lb_SA[19], ub_SA[19]],
              [lb_SA[20], ub_SA[20]],
              [lb_SA[21], ub_SA[21]],
              [lb_SA[22], ub_SA[22]],
              [lb_SA[23], ub_SA[23]],
              [lb_SA[24], ub_SA[24]],
              [lb_SA[25], ub_SA[25]],
              [lb_SA[26], ub_SA[26]],
              [lb_SA[27], ub_SA[27]],
              [lb_SA[28], ub_SA[28]],
              [lb_SA[29], ub_SA[29]],
              [lb_SA[30], ub_SA[30]],
              [lb_SA[31], ub_SA[31]],
              [lb_SA[32], ub_SA[32]],
              [lb_SA[33], ub_SA[33]]]   
    }

param_values = saltelli.sample(problem, 1024)

Y_0 = np.zeros([param_values.shape[0]])
Y_1 = np.zeros([param_values.shape[0]])
Y_2 = np.zeros([param_values.shape[0]])
Y_3 = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    # simulator_output = simulator_run(torch.tensor(X)) # run simulation
    GP_mean = model.posterior(torch.tensor(X).unsqueeze(0).to(torch.float64)).mean.flatten() # instead of performing true simulation, we estimate the output with GP posterior mean
    
    Y_0[i] = float(GP_mean[0])
    Y_1[i] = float(GP_mean[1])
    Y_2[i] = float(GP_mean[2])
    Y_3[i] = float(GP_mean[3])
    
    
Si_sensor1 = sobol.analyze(problem, Y_0)
Si_sensor2 = sobol.analyze(problem, Y_1)
Si_sensor3 = sobol.analyze(problem, Y_2)
Si_sensor4 = sobol.analyze(problem, Y_3)

print(Si_sensor1['S1'])
print(Si_sensor2['S1'])
print(Si_sensor3['S1'])
print(Si_sensor4['S1'])


import matplotlib.pyplot as plt
 
# Clean (clip negative and >1)
def clean_si(si):
    return np.clip(si, 0, 1)
 
S1, S2, S3, S4 = map(clean_si, [Si_sensor1['S1'], Si_sensor2['S1'], Si_sensor3['S1'], Si_sensor4['S1']])
 
#Stack all results
all_S = [S1, S2, S3, S4]
output_labels = ["Sensor 1: Nitrate Leaching", "Sensor 2: Water Leaching", "Sensor 3: Corn Annual Yield", "Sensor 4: Wheat Annual Yield"]
# param_labels = a.params
param_labels = ['Clay Fraction',
                'Porosity',
                'Bulk Density',
                'pH',
                'SCS Curve Number',
                'Humads C/N',
                'Humus C/N',
                'Maximum Yield (Corn)',
                'Grain C/N (Corn)',
                'Leaf C/N (Corn)',
                'Stem C/N (Corn)',
                'Root C/N (Corn)',
                'Accum. Temp. (Corn)',
                'Water Req. (Corn)',
                'Opt. Temp. (Corn)',
                'Maximum Yield (WW)',
                'Grain C/N (WW)',
                'Leaf C/N (WW)',
                'Stem C/N (WW)',
                'Root C/N (WW)',
                'Accum. Temp. (WW)',
                'Water Req. (WW)',
                'Opt. Temp. (WW)',
                'Drain Depth',
                'Drain Space',
                'Drain Factor',
                'Max N Movement',
                'Mobile N Factor',
                'Pref. N Layer Frac.',
                'N Leaching Factor',
                'N Retent. Factor',
                'Pref. N Movement',
                'Root Depth (Corn)',
                'Root Depth (Wheat)']

 
# Plot: one subplot per output
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10), sharey=True)
 
for i, ax in enumerate(axes):
    si = all_S[i]
    ax.barh(np.arange(len(si)), si, color='tab:blue')
    ax.set_title(output_labels[i])
    ax.set_xlabel("Si")
    ax.set_xlim(0, 1)  # Set x-axis range 0 to 1
 
    # Only show parameter labels on the leftmost plot
    if i == 0:
        ax.set_yticks(np.arange(len(param_labels)))
        ax.set_yticklabels(param_labels)
    else:
        ax.set_yticks(np.arange(len(param_labels)))
        ax.set_yticklabels([])

ax.set_yticklabels(param_labels)
#plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()