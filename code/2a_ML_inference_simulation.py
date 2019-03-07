# -*- coding: utf-8 -*-
"""
SUPPLEMENTARY CODE FOR BOE SWP 784: SHAPLEY REGRESSIONS: A FRAMEWORK FOR STATISTICAL INFERENCE ON MACHINE LEARNING MDOELS
-------------------------------------------------------------------------------------------------------------------------

Part 2a: ML inference simulation analysis (section 5.1 of paper):
    
    1, create config file from options
    2. loop over config file

Written for decentralised parameter sweep, e.g. on the cloud. ML_inference_from_txt()
lives independently, only relying on previous imports and the main config file.

Author: Andreas Joseph ((c) Bank of England, 2019)
"""

#%% paths and imports

print('\nML Inference: simulation')
print('------------------------\n')

main_dir = 'your/project/path/'
code_dir = main_dir+'code/'
sim_dir  = main_dir+'results/simulations/'

import os
os.chdir(code_dir)

import numpy            as     np
import pickle           as     pk
from   argparse         import Namespace  
from   ML_inference_aux import ML_inference_test, dgp_selection, nl_dgp, type_dict,\
                               ML_inference_from_txt, model_selection, model_str, reg_error,\
                               assign_line_values, write_shap_sim_config, nbr_jobs, CV_values
                               
# switch off warnings
import warnings
warnings.filterwarnings("ignore")

#%% options

run_experiment = True # if False, only write config file
verbose        = True # print results to screen
do_ML          = True # ML model on/off
do_shap        = True # Shapley value decomposition on/of
name_app       = 'test'

# parameters of experiment
DGP         = [1,2,3] # (arg: version), all processes [1,2,3]
noise_lvls  = [0.,0.1] # (arg: noise_lvl), 0 for testing
models      = ['NN','SVM','Forest'] # (arg: mdl)
# sample size is referred to as 'm' here and below, while it is 'n' in the paper
sample_size = [100,316,1000,3162,10000] # (arg: m_obs=[100] good for tesing)
k_min       = 1
k_run       = 50 # (arg: k_run=50 default)
x_ptile_cut = [5.] # (arg: x_ptile_cut=0 default)
y_ptile_cut = [5.] # (arg: y_ptile_cut=3. default)
cv_max_m    = [10000]# (arg: cv_max=1000 default)
shap_max_m  = [3162]# (arg: shap_max_m=3162 default)
min_x_frac  = [0.1]# (arg: min_x_frac=0.1 default)

# total number of jobs in simulation
n_jobs = nbr_jobs(DGP=DGP,noise_lvls=noise_lvls,models=models,sample_size=sample_size,\
                  k_run=k_run,x_ptile_cut=x_ptile_cut,y_ptile_cut=y_ptile_cut,cv_max_m=cv_max_m,\
                  shap_max_m=shap_max_m,min_x_frac=min_x_frac,verbose=verbose)

# config file
save_dir    = sim_dir+'{0}_{1}/'.format(name_app,n_jobs)
config_file = save_dir+'ML_inf_config_{0}_{1}.csv'.format(name_app,n_jobs) # file name for IO

# range to process (lines of config file)
min_read    = 1
max_read    = n_jobs # max n_jobs

#%% Part 1: create config txt file

# create output directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('Created output directory: {0}.\n'.format(save_dir))
else:
    print('Warning: Output directory already exists: {0}.\n'.format(save_dir))

# generate config file
write_shap_sim_config(DGP=DGP,noise_lvls=noise_lvls,models=models,sample_size=sample_size,\
                      type_dict=type_dict,x_ptile_cut=x_ptile_cut,y_ptile_cut=y_ptile_cut,\
                      k_run=k_run,k_min=k_min,cv_max_m=cv_max_m,shap_max_m=shap_max_m,\
                      do_ML=do_ML,do_shap=do_shap,min_x_frac=min_x_frac,file_name=config_file)
                                                
#%% Part 2: ML inference simulation (main action)

if run_experiment==True:
    for line in range(min_read,max_read+1):
        print('{0} / {1}:'.format(line,n_jobs)) # progress indicator
        ML_inference_from_txt(cfg_file=config_file,line=line,CV_values=CV_values,
                              save_dir=save_dir,ID=name_app,verbose=verbose)
                    

