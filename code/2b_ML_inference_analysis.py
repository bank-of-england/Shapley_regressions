# -*- coding: utf-8 -*-
"""
SUPPLEMENTARY CODE FOR BOE SWP 784: SHAPLEY REGRESSIONS: A FRAMEWORK FOR STATISTICAL INFERENCE ON MACHINE LEARNING MDOELS
-------------------------------------------------------------------------------------------------------------------------

Part 2b: Diagnostics on simulated ML inference results (section 5.1 of paper):
    
    1. Load single simulation or concatenated results
    2. Consistency plots: coefficients
    3. Consistency plots: error
    4. Estimation of error convergence rates

Note: The pickle 'ML_inf_joint_results_swp_final.pkl' in 'out_dir' (see below)
      includes all simulation results from the SWP. To use it, set ID='swp'
      and collect_res=False.

Author: Andreas Joseph ((c) Bank of England, 2019)
"""

print('\nML Inference: analysis')
print('----------------------\n')

#%% paths and imports
main_dir = 'your/project/path/'
code_dir = main_dir+'code/'
out_dir  = main_dir+'results/simulations/'
fig_dir  = main_dir+'figures/'

import os
os.chdir(code_dir)

import numpy             as     np
import pickle            as     pk
import pandas            as     pd
import scipy.stats       as     st
import matplotlib.pyplot as     plt
import matplotlib        as     mpl
import statsmodels.api   as     sm
from   glob              import glob
from   ML_inference_aux  import type_dict, clr_plt
from   statsmodels.sandbox.regression.predstd import wls_prediction_std

# switch off warnings
import warnings
warnings.filterwarnings("ignore")

#%% options
ID             = 'swp' # simulation ID (name ID + number of jobs)
res_dir        = out_dir+'{0}/'.format(ID) # simulation results folder
save_name      = res_dir+'ML_inf_joint_results_{0}.pkl'.format(ID) # output file name

# plot
save_plots     = True # save or not to save, that is the question
joint_plot     = False # plot single or separate plot for different coefficients
use_10_50_90   = False # use percentiles instead of mean (learning curves)
ml_conv_rate   = 0.25
fig_format     = 'pdf' # format in which to save figures
ref_var        = 'error'#'kurt'       # str or None, variabels to plot alongside learning curves (min,median,max),
# optitions: 'n_params' (not for Forest), 'error', 'model_bias', 'y_std', 'ok_fraction',
#            'f_pval', 'r2_adj', 'cond_nbr', 'JB', 'chi2-2t', 'skew', 'kurt', 'valid',
y_bounds       = None
min_frac       = 0.1 # min reconstruciton fraction
max_pval       = 0.1 # min significance level

# get configuration
os.chdir(res_dir)
df_cfg         = pd.read_csv('ML_inf_config_{0}.csv'.format(ID))
cfg_dict       = dict(zip(type_dict.keys(),[df_cfg[col].unique() for col in df_cfg.columns]))
nbr_runs       = len(cfg_dict['nbr_run'])
sample_sizes   = cfg_dict['sample_size']
n_jobs         = cfg_dict['nbr'][-1]
n_params       = {1:3,2:3,3:6} # number of parameters in each data-generating process (DGP)
collect_res    = False # collect and join single simulation results (if False need to be preloaded and stored in save_name)

#%% Part 1: Collect single simulation results

if collect_res==True:
    # main loop over configurations
    res_dict, l, k = {}, 0, 1 # k counts missing files (e.g. failed simulations), which are written to separate file
    for mdl in cfg_dict['model']:
        res_dict[mdl] = {}
        for pro in cfg_dict['DGP']:
            res_dict[mdl][pro] = {}
            for lvl in cfg_dict['noise_level']:
                res_dict[mdl][pro][lvl] = {}
                for y_cut in cfg_dict['y_cut_off']:
                    res_dict[mdl][pro][lvl][y_cut] = {}
                    for x_cut in cfg_dict['x_cut_off']:
                        res_dict[mdl][pro][lvl][y_cut][x_cut] = {}
                        # initialise
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['params']      = np.zeros((len(sample_sizes),nbr_runs,n_params[pro]))*np.nan
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['pvalues']     = np.zeros((len(sample_sizes),nbr_runs,n_params[pro]))*np.nan
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['upper']       = np.zeros((len(sample_sizes),nbr_runs,n_params[pro]))*np.nan
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['lower']       = np.zeros((len(sample_sizes),nbr_runs,n_params[pro]))*np.nan
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['params_0']    = np.zeros((len(sample_sizes),nbr_runs,n_params[pro]))*np.nan
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['upper_0']     = np.zeros((len(sample_sizes),nbr_runs,n_params[pro]))*np.nan
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['lower_0']     = np.zeros((len(sample_sizes),nbr_runs,n_params[pro]))*np.nan
                        res_dict[mdl][pro][lvl][y_cut][x_cut]['ok_fraction'] = np.zeros((len(sample_sizes),nbr_runs))*np.nan
                        if not ref_var==None:
                            res_dict[mdl][pro][lvl][y_cut][x_cut][ref_var]   = np.zeros((len(sample_sizes),nbr_runs))*np.nan
                        # iterate over sample sizes
                        for i,m in enumerate(sample_sizes):
                            for j in range(nbr_runs):
                                load_successful = False
                                # get file signature
                                f_signature = '{0}v{1}_n{2}_m{3}_xyCut_{4}-{5}_run{6}_'.format(mdl,pro,lvl,\
                                                      m,int(x_cut),int(y_cut),j+1) # X: config file line nbr (not used)
                                try: # load results pickle
                                    # line number work around
                                    valid_files = glob('*'+f_signature+'*')
                                    if len(valid_files)>1:
                                        print('\nWarning: More than one file with matching signature:\n\n\t{0}\n\nTake first one.\n'.format(f_signature))
                                        print(f_signature,valid_files)
                                    # load data
                                    out_dict = pk.load(open(valid_files[0],'rb'))
                                    load_successful = True
                                except:
                                    # record missing results
                                    print('Missing signature: {0}.'.format(f_signature))
                                    # write missing to text file
                                    file_name = out_dir+'ML_inf_config_{0}_missing.csv'.format(ID)
                                    # create line
                                    values, ln = [k,pro,lvl,mdl,m,x_cut,y_cut,j,cfg_dict['CV_max_m'],cfg_dict['shap_max_m'],cfg_dict['min_x_frac'],True,True], ''
                                    for il,v in enumerate(values):
                                        if not il==len(values)-1:
                                            ln += str(v)+','
                                        else:
                                            ln += str(v)+'\n'
                                    # write to file
                                    if k==1:
                                        hdr = ','.join(list(type_dict.keys()))+'\n'
                                        f = open(file_name,'w')
                                        f.write(hdr)
                                    f.write(ln) # write line
                                    k+=1
                                # record results content  
                                if load_successful==True:
                                    # fill in results
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['params'][i,j,:]    = out_dict['params']
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['pvalues'][i,j,:]   = out_dict['pvalues']
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['upper'][i,j,:]     = out_dict['upper']
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['lower'][i,j,:]     = out_dict['lower']
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['params_0'][i,j,:]  = out_dict['params_0']
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['upper_0'][i,j,:]   = out_dict['upper_0']
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['lower_0'][i,j,:]   = out_dict['lower_0']
                                    res_dict[mdl][pro][lvl][y_cut][x_cut]['ok_fraction'][i,j] = out_dict['ok_fraction']
                                    if not ref_var==None:
                                        res_dict[mdl][pro][lvl][y_cut][x_cut][ref_var][i,j]   = out_dict[ref_var]
                        # Learning curves
                        # ---------------
                        if joint_plot==True:
                            plt.figure(figsize=(9,6))
                            plt.title('Parameters: {0}, process {1}, {2} noise level, {3}-{4} y-x-cuts.'.format(mdl,pro,lvl,y_cut,x_cut))
                            for k in range(n_params[pro]):
                                # extract data
                                params   = res_dict[mdl][pro][lvl][y_cut][x_cut]['params'][:,:,k]
                                lower    = res_dict[mdl][pro][lvl][y_cut][x_cut]['lower'][:,:,k]
                                upper    = res_dict[mdl][pro][lvl][y_cut][x_cut]['upper'][:,:,k]
                                params_0 = res_dict[mdl][pro][lvl][y_cut][x_cut]['params_0'][:,:,k]
                                lower_0  = res_dict[mdl][pro][lvl][y_cut][x_cut]['lower_0'][:,:,k]
                                upper_0  = res_dict[mdl][pro][lvl][y_cut][x_cut]['upper_0'][:,:,k]
                                # apply filter
                                not_ok = res_dict[mdl][pro][lvl][y_cut][x_cut]['pvalues'][:,:,k]>max_pval
                                not_ok = not_ok & (res_dict[mdl][pro][lvl][y_cut][x_cut]['ok_fraction']>min_frac)
                                params[not_ok] = np.nan
                                lower[not_ok]  = np.nan
                                upper[not_ok]  = np.nan
                                # averge results (percentiles or means)
                                if use_10_50_90==True:
                                    y    = np.nanmedian(params,1)
                                    yl   = np.nanpercentile(lower,10,1)
                                    yu   = np.nanpercentile(upper,90,1)
                                    y_0  = np.nanmedian(params_0,1)
                                    yl_0 = np.nanpercentile(lower_0,10,1)
                                    yu_0 = np.nanpercentile(upper_0,90,1)
                                else:
                                    y    = np.nanmean(params,1)
                                    yl   = np.nanmean(lower,1)
                                    yu   = np.nanmean(upper,1)
                                    y_0  = np.nanmean(params,1)
                                    yl_0 = np.nanmean(lower,1)
                                    yu_0 = np.nanmean(upper,1)
                                # plot
                                plt.semilogx(sample_sizes,y,lw=3,label=r'$\beta_{0}$'.format(k))
                                if k==n_params[pro]-1:
                                    plt.axhline(y=1,lw=1,color='k',ls='--',label='ref.')
                                    plt.fill_between(sample_sizes,yl,yu,alpha=0.15,label='CI ({0})'.format(90))
                                else:
                                    plt.fill_between(sample_sizes,yl,yu,alpha=0.15)
                                # axes and labels
                                if y_bounds==None:
                                    plt.ylim([0.5,1.5])
                                else:
                                    plt.ylim(y_bounds)
                                plt.legend(loc='upper right',ncol=2)
                                plt.xlabel('sample size',fontsize=12)
                                plt.ylabel('normalised coefficient',fontsize=12)
                            # save figure
                            if save_plots==True:
                                save_name = fig_dir+'ML_inf_joint_v{0}_n{1}_xyCut_{2}-{3}_{4}.pdf'.format(pro,lvl,int(x_cut),int(y_cut),ID)
                                plt.savefig(save_name,dpi=150,bbox_inches='tight')
                            plt.show()
                            
                            # reference variable 
                            if not ref_var==None:
                                plt.figure(figsize=(9,6))
                                plt.semilogx(sample_sizes,np.nanmin(res_dict[mdl][pro][lvl][y_cut][x_cut][ref_var],1),ls='-',lw=2,   label='min')
                                plt.semilogx(sample_sizes,np.nanmedian(res_dict[mdl][pro][lvl][y_cut][x_cut][ref_var],1),ls='-',lw=2,label='mean')
                                plt.semilogx(sample_sizes,np.nanmax(res_dict[mdl][pro][lvl][y_cut][x_cut][ref_var],1),ls='-',lw=2 ,  label='max')
                                plt.title('{0}: {1}, process {2}, {3} noise level, {4}-{5} y-x-cuts (unfiltered)'.format(ref_var,mdl,pro,lvl,\
                                          y_cut,x_cut))
                                plt.legend()
                                # save figure
                                if save_plots==True:
                                    save_name = fig_dir+'ML_inf_plot_v{0}_n{1}_xyCut_{2}-{3}_ref-{4}_{5}.pdf'.format(pro,lvl,int(x_cut),int(y_cut),ref_var,ID)
                                    plt.savefig(save_name,dpi=150,bbox_inches='tight')
                                plt.show()
    # close config file for missing entries
    if k>1:
        f.close() # close file
    # save joint results
    pk.dump(res_dict,open(save_name,'wb'))
    print('\nSuccessfully wrote output to\n\n\t{0}\n'.format(save_name))
else:
    res_dict = pk.load(open(save_name,'rb'))

#%% Part 2: Coefficients convergence plots for all simulated processes, models and noise levels
                   
mdl_name_dict = {'NN':'NN','SVM':'SVM','Forest':'RF'}
# adjusted sample size for component robustness test for slow ml convergence
sample_s_adj  = np.round(sample_sizes**(2.*ml_conv_rate)).astype(int)

fs = 18 # font size
mpl.rcParams['xtick.labelsize'] = fs 
mpl.rcParams['ytick.labelsize'] = fs 

if joint_plot==False:
    for pro in cfg_dict['DGP']:
        for lvl in cfg_dict['noise_level']:
            for y_cut in cfg_dict['y_cut_off']:
                for x_cut in cfg_dict['x_cut_off']:
                    # CI correction factors
                    df_raw = sample_sizes-n_params[pro]
                    df_adj = sample_s_adj-n_params[pro]
                    df_adj_fac   = (st.t.ppf(0.95,df_adj)/st.t.ppf(0.95,df_raw))*np.sqrt(df_raw/df_adj)  # H0 adjustment
                    CI_10_raw_fac = st.t.ppf(0.55,df_raw)/st.t.ppf(0.95,df_raw)                          # H1_raw factor
                    CI_10_adj_fac = (st.t.ppf(0.55,df_adj)/st.t.ppf(0.55,df_raw))*np.sqrt(df_raw/df_adj) # H1 adjustment
                    # start figure
                    fig, axes = plt.subplots(nrows=n_params[pro], ncols=len(cfg_dict['model']), figsize=(25,40))
                    # iterate over parametres and models
                    for i in range(n_params[pro]):
                        for j,mdl in enumerate(cfg_dict['model']):
                            # extract data
                            params   = res_dict[mdl][pro][lvl][y_cut][x_cut]['params'][:,:,i]
                            lower    = res_dict[mdl][pro][lvl][y_cut][x_cut]['lower'][:,:,i]
                            upper    = res_dict[mdl][pro][lvl][y_cut][x_cut]['upper'][:,:,i]
                            params_0 = res_dict[mdl][pro][lvl][y_cut][x_cut]['params_0'][:,:,i]
                            lower_0  = res_dict[mdl][pro][lvl][y_cut][x_cut]['lower_0'][:,:,i]
                            upper_0  = res_dict[mdl][pro][lvl][y_cut][x_cut]['upper_0'][:,:,i]
                            # apply filter
                            not_ok = res_dict[mdl][pro][lvl][y_cut][x_cut]['pvalues'][:,:,i]>max_pval
                            not_ok = not_ok & (res_dict[mdl][pro][lvl][y_cut][x_cut]['ok_fraction']>min_frac)
                            params[not_ok] = np.nan
                            lower[not_ok]  = np.nan
                            upper[not_ok]  = np.nan
                            # averge results (percentiles or means)
                            if use_10_50_90==True:
                                y        = np.nanmedian(params,1)
                                # H0 raw
                                yl_0_raw = np.nanpercentile(lower,10,1)
                                yu_0_raw = np.nanpercentile(upper,90,1)
                                # H0 adjustment
                                yu_0_adj = y+(yu_0_raw-y)*df_adj_fac
                                yl_0_adj = y-(y-yl_0_raw)*df_adj_fac
                                # H1 raw
                                yu_1_raw = y+(yu_0_raw-y)*CI_10_raw_fac
                                yl_1_raw = y-(y-yl_0_raw)*CI_10_raw_fac
                                # H1 adjustment
                                yu_0_adj = y+(yu_1_raw-y)*CI_10_adj_fac
                                yl_0_adj = y-(y-yl_1_raw)*CI_10_adj_fac
                            else:
                                y        = np.nanmean(params,1)
                                # H0 raw
                                yl_0_raw = np.nanmean(lower,1)
                                yu_0_raw = np.nanmean(upper,1)
                                # H0 adjustment
                                yu_0_adj = y+(yu_0_raw-y)*df_adj_fac
                                yl_0_adj = y-(y-yl_0_raw)*df_adj_fac
                                # H1 raw
                                yu_1_raw = y+(yu_0_raw-y)*CI_10_raw_fac
                                yl_1_raw = y-(y-yl_0_raw)*CI_10_raw_fac
                                # H1 adjustment
                                yu_1_adj = y+(yu_1_raw-y)*CI_10_adj_fac
                                yl_1_adj = y-(y-yl_1_raw)*CI_10_adj_fac
                            # plot
                            #if lvl>0:
                            # lines
                            axes[i,j].semilogx(sample_sizes,y,         c='k',ls='-', lw=2.5,label=r'$\hat{\beta}(\Phi^s)$',zorder=1)
                            axes[i,j].semilogx(sample_sizes,yl_0_raw,  c='k',ls='-',lw=1,alpha=0.3,zorder=1)
                            axes[i,j].semilogx(sample_sizes,yu_0_raw,  c='k',ls='-',lw=1,alpha=0.3,zorder=1)
                            # fills
                            axes[i,j].fill_between(sample_sizes,yu_1_adj,yu_0_raw,color='b',alpha=0.15,label=r'$90\%$ CI (fast)',zorder=2)
                            axes[i,j].fill_between(sample_sizes,yl_1_adj,yu_1_adj,color='r',alpha=0.2 ,label=r'$10\%$ CI (slow)',zorder=2)
                            axes[i,j].fill_between(sample_sizes,yl_0_raw,yl_1_adj,color='b',alpha=0.15)
                            axes[i,j].axhline(y=1.0,lw=1.5, color='b',ls=':',zorder=3,label='reference')
                            axes[i,j].axhline(y=1.1,lw=1.5, color='g',ls='-.',zorder=3, label='10% bounds',alpha=0.5)
                            axes[i,j].axhline(y=0.9,lw=1.5, color='g',ls='-.',zorder=3,alpha=0.5)
                            axes[i,j].set_ylim([0.7,1.3])
                            axes[i,j].set_yticks(np.arange(0.5,1.51,0.25))
                            # legend
                            if (i==0) & (j==0):
                                axes[i,j].legend(loc='upper right',ncol=2,prop={'size':fs})
                            # model tital
                            if i==0:
                                axes[i,j].set_title(mdl_name_dict[mdl],size=fs)
                            # parameter y-label
                            if j==0:
                                axes[i,j].set_ylabel(r'$\beta_{0}$'.format(i),fontsize=fs)
                            # x-label
                            if i==(n_params[pro]-1):
                                axes[i,j].set_xlabel('sample size',fontsize=fs)
                    # save figure
                    if save_plots==True:
                        save_name = fig_dir+'ML_inf_plot_v{0}_n{1}_xyCut_{2}-{3}_{4}.pdf'.format(pro,lvl,int(x_cut),int(y_cut),ID)
                        plt.savefig(save_name,dpi=150,bbox_inches='tight')
                    plt.show()
                                
#%% Part 3: Error convergence plots for all simulated processes, models and noise levels
                    
mdl_clr_dict  = {'NN':'r','SVM':'b','Forest':'g'}
mdl_name_dict = {'NN':'NN','SVM':'SVM','Forest':'RF'}
dgp_name_dict = {1:'1 ($\gamma=2$)', 2:'1 ($\gamma=3$)', 3:'2'}

dgp_std_dict  = {1 : {0.0: 4.924, 0.1: 4.948}, 
                 2 : {0.0: 8.731, 0.1: 8.774},
                 3 : {0.0: 3.794, 0.1: 3.776}}

marker_dict = {'NN':'p','SVM':'P','Forest':'*'}

fs = 24 # font size
mpl.rcParams['xtick.labelsize'] = fs 
mpl.rcParams['ytick.labelsize'] = fs 

if (joint_plot==False) & (ref_var=='error'):
    for y_cut in cfg_dict['y_cut_off']:
        for x_cut in cfg_dict['x_cut_off']:
            # start figure
            fig, axes = plt.subplots(nrows=len(cfg_dict['DGP']), ncols=len(cfg_dict['noise_level']), figsize=(25,35))
            for i,pro in enumerate(cfg_dict['DGP']):
                for j,lvl in enumerate(cfg_dict['noise_level']):
                    # iterate over models
                    for m,mdl in enumerate(cfg_dict['model']):
                        # extract data
                        y = np.nanmean(res_dict[mdl][pro][lvl][y_cut][x_cut][ref_var],axis=1)/dgp_std_dict[pro][lvl]
                        
                        
                        # plot
                        axes[i,j].axhline(y=0,lw=1.5,color='k',ls='-',zorder=1)
                        axes[i,j].semilogx(sample_sizes,y, c=clr_plt[m+1],ls='-',
                                            marker=marker_dict[mdl],markersize=17,
                                            lw=6,label=mdl_name_dict[mdl],zorder=2)
                        # legend
                        if (i==0) & (j==0):
                            axes[i,j].legend(loc='upper right',ncol=3,prop={'size':fs})
                        # model tital
                        if i==0:
                            axes[i,j].set_title('noise level: '+str(lvl),size=fs)
                        # parameter y-label
                        if j==0:
                            axes[i,j].set_ylabel(r'DGP {0}'.format(dgp_name_dict[pro]),fontsize=fs)
                        # x-label
                        if i==(len(cfg_dict['DGP'])-1):
                            axes[i,j].set_xlabel('sample size',fontsize=fs)
            # save figure
            if save_plots==True:
                save_name = fig_dir+'ML_error_convergence_all_{0}.pdf'.format(ID)
                plt.savefig(save_name,dpi=150,bbox_inches='tight')
            plt.show()
                                                  

#%% Part 4: convergence rate estimation
fs = 14 # font size
mpl.rcParams['xtick.labelsize'] = fs 
mpl.rcParams['ytick.labelsize'] = fs 

x = np.log10(sample_sizes)

for y_cut in cfg_dict['y_cut_off']:
    for x_cut in cfg_dict['x_cut_off']:
        # start figure
        #fig, axes = plt.subplots(nrows=len(cfg_dict['DGP']), ncols=len(cfg_dict['noise_level']), figsize=(25,35))
        for i,pro in enumerate(cfg_dict['DGP']):
            for j,lvl in enumerate(cfg_dict['noise_level']):
                # iterate over models
                print()
                for mdl in cfg_dict['model']:
                    # extract data & fit
                    y   = np.log(np.nanmean(res_dict[mdl][pro][lvl][y_cut][x_cut][ref_var],axis=1)/dgp_std_dict[pro][lvl])                        
                    res = sm.OLS(y,sm.add_constant(x)).fit()
                    # print results
                    print('{0}-{1}-{2} slope: {3}, R2: {4}'.format(mdl,pro,lvl,
                          np.round(res.params[1],2),np.round(res.rsquared,2)))
                    # plot fit
                    prstd, iv_l, iv_u = wls_prediction_std(res)
                    fig, ax = plt.subplots(figsize=(8,6))
                    ax.plot(x, y, 'o', label="data")
                    ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
                    ax.plot(x, iv_u, 'r--')
                    ax.plot(x, iv_l, 'r--')
                    ax.legend(loc='best')
                    plt.show()
                    
                    