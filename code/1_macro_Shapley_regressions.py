# -*- coding: utf-8 -*-
"""
SUPPLEMENTARY CODE FOR BOE SWP 784: SHAPLEY REGRESSIONS: A FRAMEWORK FOR STATISTICAL INFERENCE ON MACHINE LEARNING MDOELS
-------------------------------------------------------------------------------------------------------------------------

Part 1: Shapley value regression analysis of UK and US macroeconomic time series (section 5.2 of paper):
    
    0. Paths, packages and options 
    1. Nested cross-validaiton
    2. Shapley values decomposition
    3. Shapley regression analysis (training and testing)
    4. Shapley stack plot (model and error decomposition)

Author: Andreas Joseph ((c) Bank of England, 2019)
"""

print('\nShapley regression analysis: macroeconomic time series')
print('------------------------------------------------------\n')

#%% paths and imports

main_dir = 'your/project/path/'
code_dir = main_dir+'code/'
data_dir = main_dir+'data/'
out_dir  = main_dir+'results/macro_time_series/'
fig_dir  = main_dir+'figures/'

import os
os.chdir(code_dir)

import numpy                    as np
import pandas                   as pd
import pickle                   as pk
import statsmodels.formula.api  as smf
import patsy                    as pt

import sklearn.model_selection  as skl_slct # need: GridSearchCV, KFold
import statsmodels.stats.outliers_influence as sm_out
import shap

from   ML_inference_aux import  model_selection, reg_error, CV_values,prep_df,\
                                model_dict, shapley_coeffs, stack_shap_plot,\
                                print_CV_results, clr_plt, PoI_dict, periods_dict,\
                                y_lim_dict, prnt_dict, handle_VAR, short_dict
                                
# switch off warnings
import warnings
warnings.filterwarnings("ignore")


#%% options
    
# model
country       = 'US' # UK or US
target        = 'Inflation' # each variable in data_file can be used
horizon       = 4 # feature target lead-lag length in quarters
mdl_type      = 'NN' # NN, Forest, SVM, Reg (OLS or Ridge, fastest for testing) or VAR (no Shapley analysis implemented)
                     # Forest needs c-extension for tree-shap. If not installed, modify lines 246+ to use kernel-shap as for NN and SVM
small_mdl     = False # include only small subset of variables
incl_lag      = True # inlclude single lag of target at horizon
case_ID       = '{0}_{1}_{2}_{3}Q_test'.format(mdl_type,country,target,horizon) # ID for saved output

# analysis steps & output
do_CV         = True # cross-validation (part 1)
do_shap       = True # Shapley decomposition (part 2)
do_reg        = True # Shapley regressions (part 3)
do_plot       = True # shap_stack_plots (part 4)
do_save       = True # save results
fig_format    = 'pdf' # format of output figure

# cross-validation (CV), training & testing
n_outer_folds = 10 # nbr outer folds for testing
n_inner_folds = 10 # nbr inner fold for CV
n_boot        = 50 # nbr bootstraps
n_train_shap  = 10 # nbr of training calculation of Shapley decomposition (10 sufficient for current analysis)
n_compos      = 4  # number of components in shap_stack plots (lower number increases clarity but omits information)
var_lags      = 1  # 1 is approximately similar to the information content received by the other models
ml_rate       = 1./3 # error convergence rate of ML models

# print basic configuration
print('country : '+country)
print('target  : '+target)
print('horizon : '+str(horizon))
print('model   : '+mdl_type)

#%% dirs and data

# directory for results
res_dir = out_dir+country+'/'+str(horizon)+'Q/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    print('Created output directory: {0}.\n'.format(res_dir))

# load data (standardised macro series)
data_file     = '{0}_macro_yoy_normed.csv'.format(country)
df            = pd.read_csv(data_dir+data_file,index_col='date',parse_dates=True)
clr_dict      = dict(zip(df.columns,clr_plt))
df, features  = prep_df(df,target,m_type=mdl_type,sft=horizon,AR=incl_lag)

# select small list of hand-picked features based on Shapley coefficients of full model
if small_mdl==True:
    case_ID  += '_short'
    features  = short_dict[country][target]
    if incl_lag==True:
        features += [target+'_{0}l'.format(horizon)]
    df = df[[target]+features]

# nbr of features and observations
n, m = len(features), len(df)

# amend colors
clr_dict['unexplained'] = clr_plt[-1]
if incl_lag==True:
    clr_dict[features[-1]] = clr_plt[-2]

#%% Part 1: Nested CV
save_name = res_dir+case_ID+'_CV_results.pkl'

if do_CV==True:
    print('\nNested CV {0}-{1}-{2}, case: {3}\n'.format(n_boot,n_outer_folds,n_inner_folds,case_ID))
    if not mdl_type=='VAR':
        # output object
        cv_dict   = {'train_pred' : np.zeros((n_boot,n_outer_folds,m))*np.nan,
                     'test_pred'  : np.zeros((n_boot,n_outer_folds,m))*np.nan,
                     'y_train'    : np.zeros((n_boot,n_outer_folds,m))*np.nan,
                     'y_test'     : np.zeros((n_boot,n_outer_folds,m))*np.nan,
                     'features'   : features,
                     'target'     : target,
                     'ID'         : case_ID,
                     'CV'         : {}}
        # outer loop
        for i in range(n_boot):
            print('Bootstrap: {0} / {1}'.format(i+1,n_boot))
            cv_dict['CV'][i] = {}
            # train-test splitting
            outer_folds = skl_slct.KFold(n_splits=n_outer_folds, shuffle=True, random_state=i).split(df[features])
            # CV over folds (inner loop)
            for j, (i_train, i_test) in enumerate(outer_folds):
                cv_dict['CV'][i][j]    = {'i_train': i_train, 'i_test': i_test}
                df_train, df_test = df.iloc[i_train], df.iloc[i_test]
                if not mdl_type in ['Reg','VAR']:
                    # cross-validation
                    inner_folds = skl_slct.KFold(n_splits=n_inner_folds, shuffle=False)
                    cv          = skl_slct.GridSearchCV(model_dict[mdl_type],CV_values[mdl_type],\
                                                        scoring='neg_mean_squared_error',cv=inner_folds)
                    cv.fit(df_train[features],df_train[target])
                    # best-CV model selection for testing
                    if mdl_type=='NN':
                        for k in cv.best_params_:
                            if type(cv.best_params_[k])==str:
                                cv.best_params_[k] = '"'+cv.best_params_[k]+'"'
                    cv_dict['CV'][i][j]['CV_vals'] = cv.best_params_
                    model = model_selection(mdl_type+'-rgr',cv.best_params_)                    
                else:
                    model = model_selection(mdl_type+'-rgr')
                
                # training
                model.fit(df_train[features],df_train[target].values.ravel())
                # save model
                if not mdl_type=='Forest':
                    cv_dict['CV'][i][j]['model'] = model
                # number of parameters (NN and SVM)
                if mdl_type=='NN':
                    cv_dict['CV'][i][j]['n_params'] = np.sum([np.prod(model.coefs_[l].shape) for l in range(model.n_layers_-1)])
                elif mdl_type=='SVM':
                    cv_dict['CV'][i][j]['n_params'] = len(model.support_)
                
                # in-sample testing
                cv_dict['train_pred'][i,j,i_train] = model.predict(df_train[features])
                cv_dict['y_train'][i,j,i_train]    = df_train[target].values
                cv_dict['CV'][i][j]['train_rmse']  = reg_error(cv_dict['y_train'][i,j,i_train],cv_dict['train_pred'][i,j,i_train])
                
                # out-of-sample testing
                cv_dict['test_pred'][i,j,i_test]   = model.predict(df_test[features])
                cv_dict['y_test'][i,j,i_test]      = df_test[target].values
                cv_dict['CV'][i][j]['test_rmse']   = reg_error(cv_dict['y_test'][i,j,i_test],cv_dict['test_pred'][i,j,i_test])
    # VAR: expanding horizon in-sample forecast
    else:
        # output object
        cv_dict   = {'train_pred' : np.zeros((1,1,m))*np.nan,
                     'test_pred'  : np.zeros((1,1,m))*np.nan,
                     'y_train'    : np.zeros((1,1,m))*np.nan,
                     'y_test'     : np.zeros((1,1,m))*np.nan,
                     'features'   : features,
                     'target'     : target,
                     'ID'         : case_ID,
                     'CV'         : {}}
        
        var_res = handle_VAR(data=df,fc_hor=horizon,max_lag=var_lags)
        cv_dict['CV'] = var_res['model']
        # fill VAR results into general CV scheme (no-OOS test, bootstrap or nested CV)
        cv_dict['train_pred'][0,0,:] = var_res['fcast']
        cv_dict['test_pred'][0,0,:]  = var_res['fcast']
        cv_dict['y_train'][0,0,:]    = df[target].values
        cv_dict['y_test'][0,0,:]     = df[target].values

    cv_dict[target] = df[target].values
    # train & test error
    cv_dict['mean_y_train'] = np.nanmean(np.nanmean(cv_dict['train_pred'],axis=1),axis=0)
    cv_dict['mean_y_test']  = np.nanmean(np.nanmean(cv_dict['test_pred'], axis=1),axis=0)
    cv_dict['train_rmse']   = reg_error(df[target].values,cv_dict['mean_y_train'], metric='RMSE')
    cv_dict['test_rmse']    = reg_error(df[target].values,cv_dict['mean_y_test'],  metric='RMSE')
    cv_dict['train_mae']    = reg_error(df[target].values,cv_dict['mean_y_train'], metric='MAE')
    cv_dict['test_mae']     = reg_error(df[target].values,cv_dict['mean_y_test'],  metric='MAE')
    cv_dict['rms_ge']       = (cv_dict['test_rmse']-cv_dict['train_rmse'])/cv_dict['test_rmse']
    cv_dict['ma_ge']        = (cv_dict['test_mae']-cv_dict['train_mae'])/cv_dict['test_mae']
    # bias-variance decompositions
    cv_dict['mse']          = reg_error(df[target].values,cv_dict['mean_y_test'],  metric='MSE')
    cv_dict['bias2']        = reg_error(df[target].values,cv_dict['mean_y_test'],  metric='bias2')
    cv_dict['var']          = (np.nanvar(np.nanmean(cv_dict['test_pred'],axis=1),axis=0).mean())
    # save results
    if do_save==True:
        pk.dump(cv_dict,open(save_name,'wb'))
# load pre-computed results (will through error if save_name not there)
else:
    cv_dict = pk.load(open(save_name,'rb'))

# print test results
print_CV_results(cv_dict)
#%% Part 2: Shapley decomposition
if mdl_type=='VAR':
    print('Shapley value decomposition not implemented for VAR.')
else:
    save_name = res_dir+case_ID+'_Shap_results.pkl'
        
    if do_shap==True:
        print('\nShapley decomposition {0}-{1}-{2}, case: {3}\n'.format(n_boot,n_outer_folds,n_inner_folds,case_ID))
        # output object
        shap_dict = {'train_shap' : np.zeros((n_boot,n_outer_folds,m,n))*np.nan,
                     'test_shap'  : np.zeros((n_boot,n_outer_folds,m,n))*np.nan,
                     'features'   : features,
                     'target'     : target, 
                     'ID'         : case_ID}
        # loop over boosttraps
        for i in range(n_boot):
            print('Bootstrap: {0} / {1}'.format(i+1,n_boot))
            # loop over test samples
            try:
                for j in range(n_outer_folds):
                    # get training and test sets
                    i_train, i_test   = cv_dict['CV'][i][j]['i_train'], cv_dict['CV'][i][j]['i_test']
                    df_train, df_test = df.iloc[i_train], df.iloc[i_test]
                    # get model
                    if mdl_type=='Forest': # forest too large to save
                        model = model_selection(mdl_type+'-rgr',cv_dict['CV'][i][j]['CV_vals'])
                        model.fit(df_train[features],df_train[target].values.ravel())
                    else:
                        model = cv_dict['CV'][i][j]['model']
                    # Shapley decomposition
                    if mdl_type=='Forest': # tree Shapley values respecting feature dependencies
                        shap_tree = shap.TreeExplainer(model,feature_dependence='tree_path_dependent')
                        shap_dict['test_shap'][i,j,i_test,:] = shap_tree.shap_values(df_test[features])
                    elif mdl_type in ['NN','SVM']:
                        shap_kern = shap.KernelExplainer(model.predict,df_train[features], link='identity')
                        if i<n_train_shap: # expensive (larger dataset)
                            shap_dict['train_shap'][i,j,i_train,:] = shap_kern.shap_values(df_train[features], l1_reg=0)
                        shap_dict['test_shap'][i,j,i_test,:] = shap_kern.shap_values(df_test[features],  l1_reg=0)
                    else:
                        shap_dict['train_shap'][i,j,i_train,:] = model.coef_*df_train[features]
                        shap_dict['test_shap'][i,j,i_test,:]   = model.coef_*df_test[features]
            except:
                print('Error: Saving intermediate results.')
                save_name = res_dir+case_ID+'_Shap_results_ERR_at_{0}.pkl'.format(i)
                pk.dump(shap_dict,open(save_name,'wb'))
                break
        # save results
        if do_save==True:
            pk.dump(shap_dict,open(save_name,'wb'))
    # load pre-computed results (will through error if save_name not there)
    else:
        shap_dict = pk.load(open(save_name,'rb'))

#%% Part 3: Shapley regressions
if mdl_type=='VAR':
    print('Shapley regression analysis not implemented for VAR.')
else:
    save_name = res_dir+case_ID+'_Reg_results.pkl'
    fml       = target+' ~ '+' + '.join(features)+' - 1'
    
    if do_reg==True:
        # output object
        reg_dict = {'ID'       : case_ID,
                    'target'   : target,  
                    'features' : list(features)}
        
        # collect feature contributions
        # -----------------------------
        shap_pnl_train = np.zeros((m*n_train_shap,n))*np.nan
        trgt_pnl_train = np.zeros(m*n_train_shap)*np.nan
        mdl_nbr_train  = np.zeros(m*n_train_shap)*np.nan
        shap_pnl_test  = np.zeros((m*n_boot,n))*np.nan
        trgt_pnl_test  = np.zeros(m*n_boot)*np.nan
        mdl_nbr_test   = np.zeros(m*n_boot)*np.nan
        pnl_time       = np.zeros(m*n_boot)*np.nan
        
        mdl_nbr  = 1
        for i in range(n_boot):
            pnl_time[i*m : (i+1)*m] = np.arange(m)
            for j in range(n_outer_folds):
                # get indices
                i_train, i_test = cv_dict['CV'][i][j]['i_train'], cv_dict['CV'][i][j]['i_test']
                # training decomposition
                if i<n_train_shap:
                    trgt_pnl_train[i*m+i_train]   = df.iloc[i_train][target].values
                    shap_pnl_train[i*m+i_train,:] = shap_dict['train_shap'][i,j,i_train,:]
                    mdl_nbr_train[i*m+i_train]    = mdl_nbr
                # test decomposition
                trgt_pnl_test[i*m+i_test]   = df.iloc[i_test][target].values
                shap_pnl_test[i*m+i_test,:] = shap_dict['test_shap'][i,j,i_test,:]
                mdl_nbr_test[i*m+i_test]    = mdl_nbr
                mdl_nbr                    += 1
                
        # put into dataframes: training results
        reg_dict['df_train']             = pd.DataFrame(data=shap_pnl_train,columns=reg_dict['features'])
        reg_dict['df_train'][target]     = trgt_pnl_train
        reg_dict['df_train']['time']     = pnl_time[:m*n_train_shap]
        reg_dict['df_train']['model_id'] = mdl_nbr_train
        # test results
        reg_dict['df_test']              = pd.DataFrame(data=shap_pnl_test,columns=reg_dict['features'])
        reg_dict['df_test'][target]      = trgt_pnl_test
        reg_dict['df_test']['time']      = pnl_time
        reg_dict['df_test']['model_id']  = mdl_nbr_test
        
        # some cleaning
        del shap_pnl_train, trgt_pnl_train, mdl_nbr_train
        del shap_pnl_test,  trgt_pnl_test,  mdl_nbr_test,pnl_time 
        
        # model averages
        ave_cols = [target]+reg_dict['features']
        reg_dict['df_ave_train'] = reg_dict['df_train'].groupby('time')[ave_cols].mean().reset_index() 
        reg_dict['df_ave_test']  = reg_dict['df_test'].groupby('time')[ave_cols].mean().reset_index()
        # set date index
        reg_dict['df_ave_train'].index = df.index
        reg_dict['df_ave_test'].index  = df.index
        
        # error decomposition
        reg_dict['df_error'] = reg_dict['df_ave_train'][features]-reg_dict['df_ave_test'][features]
        reg_dict['df_error']['unexplained'] = df[target].values-cv_dict['mean_y_train']
        reg_dict['df_error'][target]        = df[target].values
        
        
        # Shapley regressions
        # -------------------
        mode = 'test' # only for test data
        # clustered for single models
        data = reg_dict['df_'+mode]
        reg_dict['ols_all_'+mode] = smf.ols(formula=fml, data=data).fit(cov_type='nw-groupsum',
                                                    cov_kwds = {'time':    data.time.values.astype(int),
                                                                'groups':  data.model_id.values.astype(int),
                                                                'maxlags': int(len(data)**(0.25))},
                                                                use_t=True)
        # averaged by period
        for prd in periods_dict[country]:
            start, end = periods_dict[country][prd][0], periods_dict[country][prd][1]
            decomp = reg_dict['df_ave_'+mode].loc[start:end]
            data   = df.loc[start:end]
            name   = 'ols_{0}_{1}'.format(prd,mode)
            # OLS
            cov_dict       =  {'use_correction': True, 'maxlags': int(len(data)**(0.25))}
            reg_dict[name] = smf.ols(formula=fml, data=decomp).fit(cov_type='HAC',cov_kwds=cov_dict)
            
            # Shap coefficients
            se_type = 'HAC' # standard error type
            name    = 'Shap_coeffs_{0}_{1}'.format(prd,mode)
            if prd=='full': # can also other periods but quite few observations for that
                print('\n'+name.upper()+': '+case_ID)
                coeff_est = shapley_coeffs(data=data,decomp=decomp,target=target,features=features,
                                                is_TS=True,mdl_type=mdl_type,se_type=se_type,
                                                adj_for_ml_rate=False,boot_adj=False,
                                                cov_dict=cov_dict,verbose=prnt_dict[mode])
                # adjust using dof for slow ML convergence
                name_adj = 'Shap_coeffs_adj_{0}_{1}'.format(prd,mode)
                reg_dict[name_adj] = shapley_coeffs(data=data,decomp=decomp,target=target,features=features,
                                                is_TS=True,mdl_type=mdl_type,se_type=se_type,
                                                adj_for_ml_rate=True,boot_adj=False,ml_conv_rate=ml_rate,
                                                cov_dict=cov_dict,verbose=False)
                # adjust using bootstrap
                name_adj = 'Shap_coeffs_boot_{0}_{1}'.format(prd,mode)
                reg_dict[name_adj] = shapley_coeffs(data=data,decomp=decomp,target=target,features=features,
                                                is_TS=True,mdl_type=mdl_type,se_type=se_type,
                                                adj_for_ml_rate=True,boot_adj=True,ml_conv_rate=ml_rate,
                                                cov_dict=cov_dict,verbose=False)
            else: # no screen print
                coeff_est = shapley_coeffs(data=data,decomp=decomp,target=target,features=features,
                                                is_TS=True,mdl_type=mdl_type,se_type=se_type,
                                                adj_for_ml_rate=False,boot_adj=False,
                                                cov_dict=cov_dict,verbose=False)
            reg_dict[name] = coeff_est
                
            # collinearity analysis
            if (mode=='test') and (prd=='full'):
                # VIF analysis
                y, X = pt.dmatrices(fml, decomp, return_type='dataframe')
                name = 'VIF_{0}_{1}'.format(prd,mode)
                reg_dict[name] = pd.DataFrame()
                reg_dict[name]["features"] = X.columns
                reg_dict[name]["VIF"]      = [sm_out.variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                if prnt_dict[mode]==True:
                    print(name)
                    print(reg_dict[name].round(2).to_string())
        # save results
        if do_save==True:
            pk.dump(reg_dict,open(save_name,'wb'))
    # load pre-computed results (will through error if save_name not there)
    else:
        reg_dict = pk.load(open(save_name,'rb'))
        # Shapely regression results
        for mode in ['train','test']:
            # averaged by period
            for prd in periods_dict[country]:
                name      = 'Shap_coeffs_{0}_{1}'.format(prd,mode)
                if prnt_dict[mode]==True:
                    print('\n'+name.upper()+': '+case_ID)
                    print(np.round(reg_dict[name],2).to_string())

#%% Part 4: Shapley value stack plots    
if mdl_type=='VAR':
    print('Shapley value stack plot not implemented for VAR.')
else:
    if do_plot==True:
        # average test decomposition
        save_name = fig_dir+'Shap_stack_plot_{0}.{1}'.format(case_ID,fig_format)
        stack_shap_plot(df=reg_dict['df_ave_test'],target=target,features=features,
                        mdl_type=mdl_type,min_share=0,n_compos=n_compos,PoI=PoI_dict,
                        colors=clr_dict,ylim=y_lim_dict[country][target],country=country,
                        save_fig=do_save,save_name=save_name,res=350)
        # error decomposition (Shapley difference between test and training set)
        save_name = fig_dir+'Shap_stack_plot_{0}_ERR.{1}'.format(case_ID,fig_format)
        stack_shap_plot(df=reg_dict['df_error'],target=target,features=features,
                        mdl_type=mdl_type,PoI=PoI_dict,colors=clr_dict,country=country,
                        ylim=[-1.2,1.2],save_fig=do_save,save_name=save_name,
                        unexpl=True)