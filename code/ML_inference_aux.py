# -*- coding: utf-8 -*-
"""
SUPPLEMENTARY CODE FOR BOE SWP 784: SHAPLEY REGRESSIONS: A FRAMEWORK FOR STATISTICAL INFERENCE ON MACHINE LEARNING MDOELS
-------------------------------------------------------------------------------------------------------------------------

Auxilliary code for sections 5.1 and 5.2 of paper:
    
    - ML inference simulation (2a_ML_inference_simulation.py, 2b_ML_inference_analysis.py)
    - Shapley regression analysis of macro time series (1_macro_Shapley_regressions.py)
    
Function shapley_coeffs calculates Shapley share coefficients (SSC). 
    
Author: Andreas Joseph ((c) Bank of England, 2019)
"""

# packages

# basics
import os
import numpy                    as np
import pandas                   as pd
import pickle                   as pk
import scipy.stats              as st
from   argparse import Namespace  
import math

# stats & econometrics
import statsmodels.api          as sm
import statsmodels.stats.api    as sms
import statsmodels.formula.api  as smf
from   statsmodels.tsa.api      import VAR
import matplotlib.pyplot        as plt
import numpy.polynomial.polynomial as poly
import statsmodels.stats        as sm_st
import patsy                    as pt

# machine learning (from scikit-learn)
import sklearn.ensemble         as skl_ens
import sklearn.neural_network   as skl_nn
import sklearn.svm              as skl_svm
import sklearn.linear_model     as skl_lin
from   sklearn.model_selection  import GridSearchCV
import shap # shapley values calculation


# colour palette for plotting (mostly colour blind palette taken from seaborn)
clr_plt = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
           (0.6980392156862745, 0.4980392156862745, 0.7980392156862745),
           (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
           (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
           (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
           (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
           (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
           (1.0, 0.4980392156862745, 0.054901960784313725),
           (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
           (0.9254901960784314, 0.8823529411764706, 0.2),
           (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
           '#cb416b']

# types for different config settings of ML inference simulation
type_dict = {'nbr':int, 'DGP':int, 'noise_level':float, 'model':str, 'sample_size':int,
             'x_cut_off':float, 'y_cut_off':float, 'nbr_run':int, 'CV_max_m':int,
             'shap_max_m':int, 'min_x_frac':float, 'do_ML':str, 'do_shap':str}

# point-of-interest (break points)
PoI_dict      = {'1973Q2':'stagfl. start','1983Q2':'stagfl. end','2008Q3':'GFC'}

# feature lists for smaller models
short_dict    = {'UK': {'Inflation':    ['GDHI','Policy_rate','Broad_money'],
                        'GDP':          ['Policy_rate','ERI','Unemployement'],
                        'Unemployment': ['Policy_rate','GDP','Inflation']},
                 'US': {'Inflation':    ['GDP','Policy_rate','Broad_money'],
                        'GDP':          ['Policy_rate','CA','Inflation'],
                        'Unemployment': ['GDP','Inflation','Private_debt']}}


# periods corresponding to break points
periods_dict  = {'UK':{'full':    ['1956Q1','2017Q4'], 'stagflation': ['1973Q2','1983Q2'],
                       'pre-GFC': ['1983Q3','2008Q2'], 'post-GFC'   : ['2008Q3','2017Q4']},
                 'US':{'full':    ['1966Q1','2017Q4'], 'stagflation': ['1973Q2','1983Q2'],
                       'pre-GFC': ['1983Q3','2008Q2'], 'post-GFC'   : ['2008Q3','2017Q4']}}

# y bounds for shap_stack_plot 
y_lim_dict    = {'UK':{'Inflation': [-1.5,4.5], 'GDP': [-4,3.5],   'Unemployment': [-2.5,4]},
                 'US':{'Inflation': [-2,3.7],   'GDP': [-3.2,2.7], 'Unemployment': [-2.5,4]}}

# what to print to screen from Shapley regression analysis
prnt_dict     = dict(zip(['train','test'],[False,True])) 

# CV VALUES for grids search
nn0, nnf   = 3, 7 # only for NN, number of neurons per layer specification
CV_values  = {'NN':     {'hidden_layer_sizes': [(n,n)   for n in range(nn0,nn0*nnf+1,nn0)]+\
                                               [(n,n,n) for n in range(nn0,nn0*nnf+1,nn0)],
                         'alpha':              10.**np.arange(-5, 0),
                         'activation':         ['relu','tanh'],
                         'solver':             ['lbfgs'],#'adam',
                         'early_stopping':     [True]},
              'kNN':    {'n_neighbors':        list(range(1,16))},
              'Tree':   {'max_depth':          list(range(1,11))},
              'Forest': {'max_depth':          list(range(10,31,4)),
                         'n_estimators':       [200]},
              'SVM':    {'C':                  10.**np.arange(0,5),
                         'gamma':              10.**np.arange(-4,1),
                         'epsilon':            [0.,0.25,0.5,0.75]},
              'ENet':   {'alpha':              list(10.**np.arange(-4, 1)),
                         'l1_ratio':           np.arange(0,1.1,0.2),
                         'fit_intercept':      [False],
                         'max_iter':           [2000],
                         'precompute':         [True]},
              'OLS':    {'alpha':              [0]+list(10.**np.arange(-5, 1))}}
                     
# dict for model selection
model_dict = {'SVM':skl_svm.SVR(), 'NN': skl_nn.MLPRegressor(), 
              'Forest': skl_ens.RandomForestRegressor(), 'ENet': skl_lin.ElasticNet()}


def prep_df(df,target,features=None,m_type='',sft=0,AR=False,clean_data=True):
    '''Prepare df via shifts and insersion of autoregressive term. No shifte for VAR.
    
    Inputs:
    -------
    
    df : pandas dataframe
        raw input data 
        
    target : str
        name of dependent variable
   
    features : array-like (Default value = None)
         List of independent variables
         
    m_type : str  (Default value = '')
        model type name (e.g. NN for neural network). Only VAR relevant.
        
    sft : int (Default value = 0)
        lead-lag length between features and target.
        
    AR : Boolean (Default value == False)
        If to include auto-regressive feature.
        
    clean_data : Boolean (Default values = True)
        If to drop row with missing entries. 

    Outputs:
    --------
    
    length-2 list:
        [0] : transformed data
        [1] : list of independednt variables potentially including AR name
    
    '''
    
    # filter df
    if features==None:
        features = list(df.columns)
        features.remove(target)
    df = df[[target]+list(features)]
    # shif features
    if (sft>0) and not (m_type=='VAR'):
        for name in features:
            df[name] = df[name].shift(sft)
    # include AR term
    if (AR==True) and not (m_type=='VAR'):
        ar_name     = target+'_{0}l'.format(sft)
        df[ar_name] = df[target].shift(sft)
        features    = np.array(list(features)+[ar_name])
    # remove missing
    if clean_data==True:
        df.dropna(inplace=True)
    
    return [df,features]



def dgp_selection(DGP):
    '''Select parameters of data-generating process (DGP).
    
    Inputs:
    -------
    
    DGP : int
        key in [1,2,3] for data-generating polynomial process for simulation.
        
    Outputs:
    --------
    
    length-3 tuple:
        [0] : constant
        [1] : exponents of x's.
        [2] : weights of x's.
    
    '''
    
    #DGP specification
    if DGP==1: # simple model
        constant  = 0
        weights   = np.array( [2.,4.,.5,0.,0.])
        powers    = np.array([[2.,0.,0.,0.,0.],
                              [0.,1.,0.,0.,0.],
                              [0.,0.,1.,0.,0.]])
    elif DGP==2: # simple model 2
        constant  = 0
        weights   = np.array( [2.,4.,.5,0.,0.])
        powers    = np.array([[3.,0.,0.,0.,0.],
                              [0.,1.,0.,0.,0.],
                              [0.,0.,1.,0.,0.]])
    elif DGP==3: #complex model
        constant  = 2
        weights   = np.array( [2.,2.,1.,1.,.5])
        powers    = np.array([[2.,1.,1.,0.,0.],
                              [0.,1.,0.,1.,0.],
                              [0.,0.,0.,0.,1.]])
    
    return (weights,powers,constant)




def nl_dgp(x,weights,powers,noise_lvl=0,const=0):
    '''Non-linear data generating process.
    
    Inputs:
    -------
    
    x : numpy array
        2d input data of with x.shape[1]=3.
        
    weights : array-like
        weights of x components
     
    powers: numpy array
        exponent array of shape (3,len(weights))
        
    noise_lvl : float (Default value = 0)
        weight of standard Gaussian noise level scaled by standard deviation of DGP.
        
    Outputs:
    --------
    
        y : numpy array 
            simulated DGP observations 
    
    '''
    
    y = np.zeros(len(x)) # initialise
    for i,w in enumerate(weights): # loop over weights
        y += weights[i]*(x[:,0]**powers[0,i])*(x[:,1]**powers[1,i])*(x[:,2]**powers[2,i]) 
    # add noise
    if noise_lvl>0:
        y += noise_lvl*np.std(y,ddof=1)*np.random.normal(size=len(x))
        
    return y+const



def model_selection(method,val_dict=None,data=None):
    """Select model instance from scikit-learn library.

    Inputs:
    -------
    
    method : str
        model type, options: NB' (Naive-Bayes),'SVM','NN' (neural net), 'Tree','Forest','kNN','Logit','OLS','VAR' & 'ENet'
                             suffix: 'rgr' : regression problem, 'clf' : classification problem
                             NB only takes clf, No suffix for Logit, OLS, ENet and 'VAR'
                             
    val_dict : dict, optional (Default value = None)
        dictionary keyed by model parameter to be used and values to evaludated.
        if None, default values from "model_params_str" will be used.
   
    data : numpy.array, optional (Default value = None)
         needs to be given if method=='VAR' (from 'statsmodels', not 'scikit-learn').

    Outputs:
    --------
    
    model : scikit-learn model instance (VAR from statsmodels)

    """
    import sklearn.base             as skl_base
    import sklearn.ensemble         as skl_ens
    import sklearn.neural_network   as skl_nn
    import sklearn.tree             as skl_tree
    import sklearn.linear_model     as skl_lin
    import sklearn.neighbors        as skl_neigh
    import sklearn.svm              as skl_svm
    import sklearn.naive_bayes      as skl_NB
    
    # check if model choice is valid
    valid_methods = ['NN-rgr','NN-clf','Tree-rgr','Tree-clf','Forest-rgr','Forest-clf','ENet-rgr',\
                     'SVM-rgr','SVM-clf','kNN-rgr','kNN-clf','NB-clf','Reg-rgr','Reg-clf']
    if not method in valid_methods:
        raise ValueError("Invalid method: '{0}' not supported.".format(method))
    
    # build model (no VAR here)
    model = eval(model_str(method,val_dict=val_dict))
            
    return model




def model_str(method,val_dict=None):
    """Set model parameters for various ML models
    
    Inputs:
    -------
    
    method : str
        model type, options: NB' (Naive-Bayes),'SVM','NN' (neural net), 'Tree','Forest','kNN' and 'Reg'.
                             suffix: 'rgr' : regression problem, 'clf' : classification problem
    
    val_dict : dict, optional (Default value = None)
        dictionary keyed by model parameter to be used and values to evaludated.
        if None, default values from "model_params_str" will be used.
        
    Outputs:
    --------
    
    model_str : string including the model and parameter values. 

    """
    
    if method=='SVM-rgr':
        model_str    = 'skl_svm.SVR'
        all_params   = ['C','gamma','epsilon']
        default_vals = [10,0.0001,0.0] 
    elif method=='SVM-clf':
        model_str    = 'skl_svm.SVC'
        all_params   = ['C','gamma']
        default_vals = [1,0.01]
    elif method=='NN-rgr':
        model_str    = 'skl_nn.MLPRegressor'
        all_params   = ['hidden_layer_sizes','alpha','activation','solver','max_iter']
        default_vals = [(2,2), 0.1,'"tanh"','"lbfgs"',2000] 
    elif method=='NN-clf':
        model_str    = 'skl_nn.MLPClassifier'
        all_params   = ['hidden_layer_sizes','alpha','activation','solver']
        default_vals = [(10,),0.001,'"tanh"','"lbfgs"',2000]
    elif method=='Tree-rgr':
        model_str    = 'skl_tree.DecisionTreeRegressor'
        all_params   = ['max_depth']
        default_vals = [2]
    elif method=='Tree-clf':
        model_str    = 'skl_tree.DecisionTreeClassifier'
        all_params   = ['max_features','max_depth']
        default_vals = ['"sqrt"',5]
    elif method=='Forest-rgr':
        model_str    = 'skl_ens.RandomForestRegressor'
        all_params   = ['n_estimators','max_depth']
        default_vals = [200,8] 
    elif method=='Forest-clf':
        model_str    = 'skl_ens.RandomForestClassifier'
        all_params   = ['n_estimators','criterion','max_features','max_depth']
        default_vals = [200,'"entropy"','"sqr"t',10]
    elif method=='kNN-rgr':
        model_str    = 'skl_neigh.KNeighborsRegressor'
        all_params   = ['n_neighbors','p']
        default_vals = [2,2] 
    elif method=='kNN-clf':
        model_str    = 'skl_neigh.KNeighborsClassifier'
        all_params   = ['n_neighbors','p']
        default_vals = [1,2]
    elif method=='Reg-rgr':
        model_str    = 'skl_lin.Ridge'
        all_params   = ['alpha','fit_intercept']
        default_vals = [0,False]
    elif method=='ENet-rgr':
        model_str    = 'skl_lin.ElasticNet'
        all_params   = ['alpha','l1_ratio','fit_intercept']
        default_vals = [0.,0.,True]
    elif method=='Reg-clf':
        model_str    = 'skl_lin.logistic.LogisticRegression'
        all_params   = ['C']
        default_vals = [100]
    else:
        raise ValueError('Model default parameter values not given.')
    
    # construct paramter string
    params_str   = ''        
    for i,p in enumerate(all_params):
        if not val_dict==None:
            if p in val_dict.keys():
                d = val_dict[p]
            else:
                d = default_vals[i]
        else:
            d = default_vals[i]
        params_str += p+'='+str(d)+','
    
    # join model and parameters    
    model_str +=  '({0})'.format(params_str[:-1]) # remove last comma
    #print model_str
    
    return model_str


def handle_VAR(data,fc_hor=1,max_lag=1):
    '''VAR needs to be handled differently in current train-test setting.
       (not 100% comparable with other modes): in-sample forecasting test.
    
    Inputs:
    -------
    
        data : numpy array
            VAR input data data.shape[0] = n obs.
            
        fc_hor : int (Default value = 1)
            forecasting horizon in data steps (rows)
            
        max_lag : int (Default value = 1)
            number of lags in VAR

    Ouputs:
    -------
        output dictionary : model, forcasted values and 
                            focrecast-appended input data
    '''
    
    # model fit
    model = VAR(data).fit(maxlags=max_lag)
    
    # in-sample test forcasts
    fcast = np.zeros(len(data))*np.nan
    # iterate over forecast points
    for i in range(fc_hor+max_lag-1,len(data)):
        fcast[i] = model.forecast(data.values[i-fc_hor-max_lag+1:i-fc_hor+1,:],fc_hor)[-1,0]
    data['fcast'] = fcast
    
    return {'model' : model,
            'fcast' : fcast,
            'data'  : data.copy()}
    


def reg_error(y_ref,y_pred,metric='RMSE',n_digit=None,agg=False):
    '''Calculate error for regression problem.
    
    Inputs:
    -------
    
        y_ref : array-like
            reference values
        
        y_pred : array-like
            predicted values
            
        metric : str (Default value = RMSE)
            error metric choice (RMSE, MAE, MSE, bias2 or var)
            
        n_digit : int (Default value = None)
            number of digits output is rounded to
            
        agg : Boolean (Default value = False)
            If to average values over observations.
            
    Outputs:
    --------
    
        error : float
    
    '''
    
    if agg==True: # hard-coded to current case
        y_pred = np.nanmean(np.nanmean(y_pred,axis=1),axis=0)
        if   metric=='RMSE': # mean squared error
            diff  = np.abs(np.array(y_ref)-np.array(y_pred))
            error = np.sqrt(np.nanmean(diff**2))
        elif metric=='MAE': # mean absolute error
            diff  = np.abs(np.array(y_ref)-np.array(y_pred))
            error = np.nanmean(diff)
        elif metric=='MSE': # bias squared
            error = np.nanmean(np.nanmean((y_ref-y_pred)**2,axis=1),axis=0).mean()
        elif metric=='bias2': # mean absolute error
            mse   = np.nanmean(np.nanmean((y_ref-y_pred)**2,axis=1),axis=0).mean()
            error = (np.nanmean(np.nanmean(np.abs(y_ref-y_pred),axis=1),axis=0).mean()**2)/mse
        elif metric=='var': # model variance
            mse   = np.nanmean(np.nanmean((y_ref-y_pred)**2,axis=1),axis=0).mean()
            error = (np.nanvar(np.nanmean(np.abs(y_pred),axis=1),axis=0).mean())/mse
    # 1d series
    else:   
        diff = np.abs(np.array(y_ref)-np.array(y_pred))
        if   metric=='RMSE': # mean squared error
            error = np.sqrt(np.nanmean(diff**2))
        elif metric=='MAE': # mean absolute error
            error = np.nanmean(diff)
        elif metric=='MSE': # mean squared error
            error = np.nanmean(diff**2)
        elif metric=='bias2': # bias squared
            error = np.nanmean(diff)**2
        elif metric=='var': # model variance
            error = np.nanvar(y_pred)
        else:
            raise ValueError('Invalid error metric.')
        
    if not n_digit==None:
        error = np.round(error,n_digit)
        
    return error


def assign_line_values(row,type_dict):
    '''Values assignment between dict keys and row values by dict value types
    
    Inputs:
    -------
    
        row : array-like
            values to assign
            
        type_dict : dict
            dictionary keyed by variables names and values by variable dtypes
            
    Outputs:
    --------
    
        out_dict : dict
            dictionary keyed by type_dict.keys() and values from row. 
    
    '''
    
    if not len(row)==len(type_dict):
        raise ValueError('Row data and names not matching.')
    else:
        out_dict = {}
        for val,name in zip(row,type_dict.keys()):
            if type_dict[name]==str:
                exec('out_dict["{0}"] = "{1}"'.format(name,val))
            else:
                exec('out_dict["{0}"] = {1}'.format(name,val))
            
        return out_dict
            
    


def ML_inference_test(DGP,noise_lvl,mdl,m_obs,k_run,x_ptile_cut=0,y_ptile_cut=3.,\
                      do_ML=True,CV_values=None,cv_max_m=1000,do_shap=True,shap_max_m=3162,\
                      shap_part=10,min_x_frac=0.1,verbose=False):   
    '''Test statistical inference of ML models on simulated polynomial data-generating processes (DGP).
    
    Inputs:
    -------
    
        DGP : int
            ID of DGP
        
        noise_lvl : float
            noise level of DGP
        
        mdl : str
            model type
        
        m_obs : int
            number of observations
        
        k_run : int
            number of independent runs for robustness
        
        x_ptile_cut : int (Defaults value = 0)
            outlier cutoff for reconstructed x-values
        
        y_ptile_cut : int (Defaults value = 0)
            outlier cutoff for reconstructed y-values
        
        do_ML : Boolean (Defaults value = True)
            If to do machine learning model calibration and fit
        
        CV_values : dict (Defaults value = None)
            Cross-validation(CV) values for grid search of hyperparameters
        
        cv_max : int (Defaults value = 1000)
            max number of observations for CV
        
        do_shap : Boolean (Defaults value = True)
            If to do Shapley value calculation
        
        shap_max_m : int (Defaults value = int(10**3.5))
            maximal size of background sample for shapley value calculation
        
        shap_part : int (Defaults value = 10)
            number of samples for Shapley value calculation
        
        min_x_frac : float (Defaults value = 0.1)
            minimal fraction of validly reconstructed x values from Shapley values
        
        verbose : Boolean (Defaults value = False)
            If intermediate informaiton is printed from process simulation
        
    Outputs:
    --------
    
        res_dict : dictionary containing simulation results (content explained by key names)
    
    '''
    
    # results container
    res_dict = {'v': DGP, 'noise': noise_lvl, 'model': mdl, 'm_obs': int(m_obs), 'k': k_run,\
                'y_cut': y_ptile_cut, 'x_cut': x_ptile_cut, 'CV_vals': [], 'models': []}
    # process selection
    weights, powers, constant = dgp_selection(DGP)
    
    features  = ['f1','f2','f3']
    target    = ['y']

    # generate syntetic data
    m          = res_dict['m_obs']
    X_cv       = np.random.normal(size=(m,len(features)))
    X_train    = np.random.normal(size=(m,len(features)))
    X_test     = np.random.normal(size=(m,len(features)))
        
    df_cv      = pd.DataFrame(data = np.hstack((X_cv,   nl_dgp(X_cv,   weights,powers,noise_lvl,const=constant).reshape((m,1)))), columns=features+target)
    df_train   = pd.DataFrame(data = np.hstack((X_train,nl_dgp(X_train,weights,powers,noise_lvl,const=constant).reshape((m,1)))), columns=features+target)
    df_test    = pd.DataFrame(data = np.hstack((X_test, nl_dgp(X_test, weights,powers,noise_lvl,const=constant).reshape((m,1)))), columns=features+target)
    
    # Linear model: theoretical confidence boounds with finite noise level
    x1, x2, x3, y_test = df_test.values[:,0], df_test.values[:,1], df_test.values[:,2], df_test[target].values[:,0]
    w1, w2, w3, w4, w5 = weights
    if noise_lvl>0:
        if   DGP==1:
            t1, t2, t3 = (w1*x1**2).reshape((m,1)), (w2*x2).reshape((m,1)), (w3*x3).reshape((m,1))
            compos     = ['p1','p2','p3','y']
            X_org      = np.hstack((t1,t2,t3))
        elif DGP==2:
            t1, t2, t3 = (w1*x1**3).reshape((m,1)), (w2*x2).reshape((m,1)), (w3*x3).reshape((m,1))
            compos     = ['p1','p2','p3','y']
            X_org      = np.hstack((t1,t2,t3))
        elif DGP==3:
            # get polynomial terms
            t1, t2, t3, t4, t5 = (w1*x1**2).reshape((m,1)), (w2*x1*x2).reshape((m,1)), (w3*x1).reshape((m,1)),\
                                 (w4*x2).reshape((m,1)),    (w5*x3).reshape((m,1))
            compos = ['p1','p2','p3','p4','p5','y']
            X_org  = np.hstack((t1,t2,t3,t4,t5))
        # Shapley regression
        df_X = pd.DataFrame(data = np.hstack((X_org,y_test.reshape((m,1)))),columns = compos) # data frame of reconstracted X
        # ols fit
        if not (constant==0):
            ols_0 = sm.OLS(df_X[compos[-1]],sm.add_constant(df_X[compos[:-1]])).fit(cov_type='HC3')
        else:
            ols_0 = sm.OLS(df_X[compos[-1]],                df_X[compos[:-1]] ).fit(cov_type='HC3')
        # record results
        res_dict['params_0']  = ols_0.params.values
        res_dict['pvalues_0'] = ols_0.pvalues.values
        res_dict['lower_0']   = ols_0.conf_int(alpha=0.05)[0].values
        res_dict['upper_0']   = ols_0.conf_int(alpha=0.05)[1].values
    else:
        for name in ['params_0','pvalues_0','lower_0','upper_0']:
            res_dict[name] = np.nan
    
    # ML inference analysis
    if do_ML=='True':
        # for model selection
        model_dict = {'SVM':skl_svm.SVR(), 'NN': skl_nn.MLPRegressor(), 'Forest': skl_ens.RandomForestRegressor()}
        
        # Cross-validation:
        # -----------------
        if not CV_values==None:
            
            # grid search    
            cv = GridSearchCV(model_dict[mdl],CV_values[mdl],scoring='neg_mean_squared_error',verbose=0,cv=2)
            if mdl=='Forest': # no CV size restriction
                cv.fit(df_cv.loc[:,features],df_cv.loc[:,target].values.ravel()) # get best_params_
            else:
                cv.fit(df_cv.loc[:np.min([m,cv_max_m]),features],df_cv.loc[:np.min([m,cv_max_m]),target].values.ravel())
            # use CV results to training and testing
            if mdl=='NN':
                for k in cv.best_params_:
                    if type(cv.best_params_[k])==str:
                        cv.best_params_[k] = '"'+cv.best_params_[k]+'"'
            res_dict['CV_vals'].append(cv.best_params_) # save
            model = model_selection(mdl+'-rgr',cv.best_params_)
        else:
            model = model_selection(mdl+'-rgr') # default values
        
        # training
        # --------
        model.fit(df_train[features],df_train[target].values.ravel())
        if not mdl=='Forest':
            res_dict['models'].append(model) # save
        # number of parameters (NN and SVM)
        if mdl=='NN':
            res_dict['n_params'] = np.sum([np.prod(model.coefs_[l].shape) for l in range(model.n_layers_-1)])
        elif mdl=='SVM':
            res_dict['n_params'] = len(model.support_)
        
        # testing
        # -------
        y_pred = model.predict(df_test[features])
        y_diff = y_test-y_pred
        res_dict['error']      = reg_error(y_test,y_pred)
        res_dict['model_bias'] = np.mean(np.abs(y_diff))
        res_dict['y_std']      = np.std(y_test,ddof=1)
        
        # feature attribution (Shapley values)
        # ------------------------------------
        shap_keys = ['ok_fraction','valid','params','pvalues','lower','upper',
                     'f_pval','r2_adj','cond_nbr','JB','chi2_2t','skew','kurt']
        if do_shap=='True':
            if m<=shap_max_m:
                shap_kern = shap.KernelExplainer(model.predict, df_train[features],link="identity")
                attr_vals = shap_kern.shap_values(df_test[features],link="identity",l1_reg=0)
            else:
                # iterate over shap_part index partitions
                attr_vals   = np.zeros(df_test[features].shape)
                i_test_part = rnd_idx_part(m,shap_part)
                for i,prt in enumerate(i_test_part):
                    print('Shapley decomposition for partition {0} / {1}:'.format(i+1,shap_part))
                    shap_kern        = shap.KernelExplainer(model.predict, df_train[features].iloc[np.random.choice(m,shap_max_m),:],link="identity")
                    attr_vals[prt,:] = shap_kern.shap_values(df_test[features].iloc[prt,:],link="identity",l1_reg=0)
                    print('{0} values inderted, fraction remaining {1}.\n'.format(len(prt),np.round(np.sum(attr_vals[:,0]==0)/float(m),3)))
            
            # recover components from shapley values
            mdl_mean   = y_pred.mean()
            s1, s2, s3 = attr_vals[:,0], attr_vals[:,1], attr_vals[:,2]
            if DGP==1:
                xs1       = ((s1+mdl_mean)/w1)**(1./2) # sqrt
                xs1[x1<0] = -xs1[x1<0]
                xs2 = s2/w2
                xs3 = s3/w3
                t1, ts1 = (w1*x1**2).reshape((m,1)), (w1*xs1**2).reshape((m,1))
                t2, ts2 = (w2*x2).reshape((m,1)),    (w2*xs2).reshape((m,1))
                t3, ts3 = (w3*x3).reshape((m,1)),    (w3*xs3).reshape((m,1))
                compos  = ['p1','p2','p3','y']
                X_org   = np.hstack((t1,t2,t3))
                X_rec   = np.hstack((ts1,ts2,ts3))
            elif DGP==2:
                xs1       = ((s1+mdl_mean)/w1)**(1./3) # 3-root
                xs1[x1<0] = -xs1[x1<0]
                xs2 = s2/w2
                xs3 = s3/w3
                t1, ts1 = (w1*x1**3).reshape((m,1)), (w1*xs1**3).reshape((m,1))
                t2, ts2 = (w2*x2).reshape((m,1)),    (w2*xs2).reshape((m,1))
                t3, ts3 = (w3*x3).reshape((m,1)),    (w3*xs3).reshape((m,1))
                compos  = ['p1','p2','p3','y']
                X_org   = np.hstack((t1,t2,t3))
                X_rec   = np.hstack((ts1,ts2,ts3))
            elif DGP==3:
                s12 = s1+s2+mdl_mean
                # solve for original x
                xs1   = (-(w2*x2+w3)+np.sqrt((w2*x2+w3)**2-4*w1*(w4*x2-s12+constant)))/(2*w1)
                xs1_m = (-(w2*x2+w3)-np.sqrt((w2*x2+w3)**2-4*w1*(w4*x2-s12+constant)))/(2*w1) # negative branch
                xs1[x1<0] = xs1_m[x1<0]
                xs2   = (s12-w1*x1**2-w3*x1-constant)/(w2*x1+w4)
                xs3   = s3/w5
                # get polynomial terms
                t1, ts1 = (w1*x1**2).reshape((m,1)), (w1*xs1**2).reshape((m,1))
                t2, ts2 = (w2*x1*x2).reshape((m,1)), (w2*xs1*xs2).reshape((m,1))
                t3, ts3 = (w3*x1).reshape((m,1)),    (w3*xs1).reshape((m,1))
                t4, ts4 = (w4*x2).reshape((m,1)),    (w4*xs2).reshape((m,1))
                t5, ts5 = (w5*x3).reshape((m,1)),    (w5*xs3).reshape((m,1))
                compos  = ['p1','p2','p3','p4','p5','y']
                X_org   = np.hstack((t1,t2,t3,t4,t5))
                X_rec   = np.hstack((ts1,ts2,ts3,ts4,ts5))
            
            # cleaning
            rec_ratio        = X_rec/X_org
            lower_y, upper_y = np.nanpercentile(y_diff,y_ptile_cut/2), np.nanpercentile(y_diff,100-y_ptile_cut/2)
            is_ok = ((y_diff >= lower_y) & (y_diff <= upper_y)) & (~np.isnan(X_rec.sum(1)))
            for c in range(rec_ratio.shape[1]):
                r     = rec_ratio[:,c]
                is_ok = is_ok & ((r >= np.nanpercentile(r,x_ptile_cut/2)) & (r <= np.nanpercentile(r,100-x_ptile_cut/2)))
            res_dict['ok_fraction'] = float(np.sum(is_ok))/len(is_ok)
            
            # Shapley regression
            df_rec = pd.DataFrame(data = np.hstack((X_rec,y_test.reshape((m,1)))),columns = compos) # data frame of reconstracted X
            if res_dict['ok_fraction']>min_x_frac: # minimum reconstruction fraction
                res_dict['valid'] = True
                # ols fit
                if not (constant==0):
                    ols = sm.OLS(df_rec.loc[is_ok,target],sm.add_constant(df_rec.loc[is_ok,compos[:-1]])).fit(cov_type='HC3')
                else:
                    ols = sm.OLS(df_rec.loc[is_ok,target],df_rec.loc[is_ok,compos[:-1]]).fit(cov_type='HC3')
                # record results
                res_dict['params']  = ols.params.values
                res_dict['pvalues'] = ols.pvalues.values
                res_dict['lower']   = ols.conf_int(alpha=0.05)[0].values
                res_dict['upper']   = ols.conf_int(alpha=0.05)[1].values
                # test statistics
                res_dict['f_pval'], res_dict['r2_adj'],  res_dict['cond_nbr'] = ols.f_pvalue, ols.rsquared_adj, ols.condition_number
                res_dict['JB'],     res_dict['chi2_2t'], res_dict['skew'], res_dict['kurt'] = sms.jarque_bera(ols.resid)
                # set constant to one
                if not (constant==0):
                    res_dict['params'][0] /= constant
                    res_dict['lower'][0]  /= constant
                    res_dict['upper'][0]  /= constant
                # print summaries
                if verbose==True:
                    print(ols.summary())
            # bad process reconstruction
            else:
                res_dict['valid'] = False
                for name in shap_keys[2:]:
                    res_dict[name] = np.nan
        else:
            for name in shap_keys:
                res_dict[name] = np.nan

    return res_dict


def ML_inference_from_txt(cfg_file,line,CV_values,save_dir,ID='',verbose=True):
    '''ML inference based on single line from config file.'''
    
    if os.path.exists(cfg_file):
        # iterate over lines in csv and forwards parameters to inference test function
        for l,ln in enumerate(open(cfg_file)):
            if line==l: # filter line
                # unpack values to sub-namespace
                d = Namespace(**assign_line_values(ln[:-1].split(','),type_dict))
                # main action
                res_dict = ML_inference_test(DGP=d.DGP, noise_lvl=d.noise_level, mdl=d.model,\
                                             m_obs=d.sample_size, k_run=d.nbr_run,\
                                             x_ptile_cut=d.x_cut_off, y_ptile_cut=d.y_cut_off,\
                                             CV_values=CV_values,cv_max_m=d.CV_max_m,\
                                             shap_max_m=d.shap_max_m,min_x_frac=d.min_x_frac,\
                                             do_ML=d.do_ML,do_shap=d.do_shap,verbose=verbose)
                # save results
                save_name = save_dir+'ML_inf_{8}_{0}v{1}_n{2}_m{3}_xyCut_{4}-{5}_run{6}_{7}.pkl'.format(d.model,d.DGP,\
                                                  d.noise_level,int(d.sample_size),int(d.x_cut_off),int(d.y_cut_off),\
                                                  d.nbr_run,ID,l)
                if not os.path.exists(save_name):
                    pk.dump(res_dict,open(save_name,'wb'))
                    print('\nSuccessfully wrote output to {0}\n'.format(save_name))
                else:
                    print('Warning: Results not saved. File already exists:\n\n\t{0}.\n\n'.format(save_name))
    else:
        print('\nSimulation aborted. Config file does not exist: {0}.\n'.format(cfg_file))
        
        

def write_shap_sim_config(DGP,noise_lvls,models,sample_size,type_dict,file_name,\
                          x_ptile_cut=[5.],y_ptile_cut=[0.],k_run=50,k_min=1,\
                          cv_max_m=[1000],do_ML=True,do_shap=True,shap_max_m=[1000],\
                          min_x_frac=[0.1]):
    '''Write config file for ML inference simulation.
    
    Inputs:
    -------
    
        DGP : int
        ID of DGP
        
        noise_lvls : array-like
            noise levels of simulated DGPs
        
        models : array-like
            model types to simulate
        
        sample_size : array-like
            sample sizes to be simulated
            
        type_dict : dict
            dictionary keyed by variables names and values by variable dtypes
            
        file_name : str
            name config file to be saved
        
        k_run : int (Defaults value = 50)
            max run label and number of independent runs for robustness for each configuration
            
        k_min : int (Defaults value = 1)
            min run label
        
        x_ptile_cut : array-like (Defaults value = [5.])
            outlier cutoffs for reconstructed x-values to be simulated
        
        y_ptile_cut : array-like (Defaults value = [5.])
            outlier cutoffs for reconstructed y-values to be simulated
        
        do_ML : Boolean (Defaults value = True)
            If to do machine learning model calibration and fit
        
        do_shap : Boolean (Defaults value = True)
            If to do Shapley value calculation
        
        shap_max_m : array-like (Defaults value = [1000])
            maximal size of background sample for shapley value calculation to be simulated
        
        min_x_frac : to be simulated (Defaults value = [0.1])
            minimal fraction of validly reconstructed x values from Shapley values
            
        Outputs:
        --------
        
            None : writes config file
    
    '''
    
    if not os.path.exists(file_name):
        # number of lines in config file (one per job)
        n_jobs = nbr_jobs(DGP=DGP,noise_lvls=noise_lvls,models=models,sample_size=sample_size,\
                  k_run=k_run,x_ptile_cut=x_ptile_cut,y_ptile_cut=y_ptile_cut,cv_max_m=cv_max_m,\
                  shap_max_m=shap_max_m,min_x_frac=min_x_frac,verbose=False)
        # header
        hdr = ','.join(list(type_dict.keys()))+'\n'
        # loop through parameter space
        k = 1 # line number
        # oder needs to match keys in type_dict
        for v1 in DGP:
            for v2 in noise_lvls:
                for v3 in models:
                    for v4 in sample_size:
                        for v5 in x_ptile_cut:
                            for v6 in y_ptile_cut:
                                for v7 in range(k_min,k_min+k_run):
                                    for v8 in cv_max_m:
                                        for v9 in shap_max_m:
                                            for v10 in min_x_frac:
                                                # create line
                                                values, ln = [k,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,do_ML,do_shap], ''
                                                for i,v in enumerate(values):
                                                    if not i==len(values)-1:
                                                        ln += str(v)+','
                                                    else:
                                                        ln += str(v)+'\n'
                                                # write to file
                                                if k==1:# create file and write header
                                                    f = open(file_name,'w')
                                                    f.write(hdr)
                                                f.write(ln) # write line
                                                k+=1 
        f.close() # close file
        print('Successfully created config file ({0} rows):\n\n\t{1}\n\n'.format(n_jobs,file_name))
    else:
        print('Warning: Config not written. File already exists:\n\n\t{0}.\n\n'.format(file_name))
        
        

def nbr_jobs(DGP,noise_lvls,models,sample_size,k_run=50,x_ptile_cut=[5.],y_ptile_cut=[0.],\
             cv_max_m=[1000],shap_max_m=[1000],min_x_frac=[0.1],verbose=False):
    '''Calculate total number of jobs for parameter sweep. 
    
    Check write_shap_sim_config() for input definitions.
    
    Outputs:
    --------
    
        n_jobs : int
            number of iterations for certain configuration
    
    '''

    n_jobs  = len(DGP)*len(noise_lvls)*len(models)*len(sample_size)*k_run*len(x_ptile_cut)
    n_jobs *= len(y_ptile_cut)*len(cv_max_m)*len(shap_max_m)*len(min_x_frac)
    
    if verbose:
        print('\nTotal number of jobs (parameter configurations & iterations): {0}.\n'.format(n_jobs))
    
    return n_jobs


def rnd_idx_part(m,n):
    '''n (int) approx equally-sized random particitions of range(m) index.'''
    
    d, lst = m/float(n), np.arange(m) # nbr of partitions and index range
    np.random.shuffle(lst) # shuffle index range
    # nested list of random partitions 
    idx_part = [ list(lst[int(round(d*i)):int(round(d*(i+1)))]) for i in range(n) ]
    
    return idx_part


def shapley_coeffs(data,decomp,target,features,mdl_type,is_TS=False,se_type='H3',cov_dict=None,\
                   adj_for_ml_rate=False,ml_conv_rate=1./3,boot_adj=False,n_boot=100,\
                   bias_conf_lvl=0.80,sort_share=True,verbose=False,intercept=True):
    '''Calculate shapley share coefficients for Shapley regression. Heteroscedasticity and robust regressions is 
       dealt with by "se_type" and "cov_dict" depending on application. "adj_for_ml_rate" accounts for 
       slow ML convergence rate assuring asymptotic unbiasedness.
    
    Inputs:
    -------
    
        data   : pandas dataframe
            Original input data of dimension (nbr observations, nbr features)
    
        decomp : pandas dataframe
            Shapley values decomposition of model outputs of dimension (nbr observations, nbr features)
            
        target: str
            model dependent variable
            
        features : array-like
            model independent variables
            
        mdl_type : str
            model type (e.g. NN or Reg). Linear regression model (Reg) treated differently (exact calculation)
            
        is_TS : Boolean (Default value = False)
            Wheter data has time series dimension or not for standard error calculation
            
        se_type : str (Default value = 'H3')
            type of regression standard error. Needs to be compatible with statsmodels OLS
            
        cov_dict : dict (Default value = None)
            options for regression standard error calculation. Needs to be compatible with statsmodels OLS
            
        adj_for_ml_rate : Boolean (Default value = True)
            Whether to adjust for the slower rate of convergence of ML/non-parametric model. 
            
        ml_conv_rate : float (Default value = 1./3)
             ML error convergence rate proportional to r**(-ml_conv_rate).
             Dof adjuctment for hypothesis testing proportional to sample_sizes**(2*ml_conv_rate).
             
        boot_adj : Boolean (Default value = False)
            Whether to adjust ML dof difference using bootstrapped samples 
            
        n_boot : int (Default values = 100)
            Number of bootstraps used for bootstrap inference adjustment (if not is_TS==True)
            
        bias_conf_lvl : float (Default value = 0.68)
            confidence level for unbiased component (one needs to be within 1-bias_conf_lvl CI of coefficient)
            
        sort_share : Boolean (Default value = True)
            Whether to sort output by absolute Shapley value decomposition share
            
        verbose : Boolean (Default value = False)
            Whether to print resulting dataframe to screen
            
        intercept : Boolean (Default value = True)
            Whether to include intercept in Shapley regression
    
    Output:
    -------
    
        pandas dataframe of shapely coefficients and regression stats
        
    '''
    # reassign original column names
    col_names = features.copy()

    # standardise feature names
    feat_indx = np.arange(1, len(features) + 1).astype(str)
    features = ['x' + f for f in feat_indx]
    data.columns = np.append('y', features)

    # Create dictionary b/w old and new column name
    col_indx = np.append('Intercept', features)
    col_names = np.append('Intercept', col_names)
    col_zip = zip(col_indx, col_names)
    col_dict = dict(col_zip)

    # column bind response variable and SHAP values and convert to dataframe
    shap_val = np.column_stack((data.iloc[:,0], decomp))
    shap_val = pd.DataFrame(shap_val)
    shap_val.columns = np.append('y', features)
    decomp = shap_val

    # regression formula
    fml = target+' ~ '+' + '.join(features)
    if intercept==False: # remove intercept
        fml += ' - 1'
    n = len(features)
    # for test of constraint model (feature robustness)
    R = np.concatenate((np.identity(n),np.ones((1,n))),axis=0) # constraints matrix:last row corresponds to 
                                                               # sum of coefficients equals n
    if intercept==True:
        R = np.concatenate((np.zeros((n+1,1)),R),axis=1)
        q = np.append(n,np.ones(n)) 
    else:
        q = np.ones(n)
    CI = 'w_one_'+str(int(100*bias_conf_lvl)) # column name for confidence interval for beta^S
        
    # shapley regression
    rgr_ols = smf.ols(formula=fml, data=data).fit(cov_type=se_type,cov_kwds=cov_dict,use_t=True) # standard regression
    if mdl_type in ['Reg','ENet']: # standard regression for linear model
        rgr_shap = rgr_ols # keep originat coefficient estimates
        decomp[features] = rgr_shap.params[features]*data[features] # Shapley effect sizes
    else: 
        rgr_shap = smf.ols(formula=fml, data=decomp).fit(cov_type=se_type,cov_kwds=cov_dict,use_t=True)
        rgr_shap.pvalues = rgr_shap.pvalues/2 # for one-sided test
        # adjust Dof for slower convergence of ML estimators for valid inference
        if ml_conv_rate>=0.5: # no adjustment needed
            adj_for_ml_rate = False
        if adj_for_ml_rate==True:
            if boot_adj==False: # DoF adjustment
                df_raw = len(data)-int(rgr_shap.df_model)
                df_adj = np.round(len(data)**(2.*ml_conv_rate)).astype(int)-int(rgr_shap.df_model) # dof
                se_adj = rgr_shap.bse*np.sqrt(df_raw/df_adj) # standard errors
                rgr_shap.tvalues  = rgr_shap.params/se_adj # adjusted t-values
                rgr_shap.pvalues  = 1-st.t.cdf(np.abs(rgr_shap.tvalues),df_adj) # adjusted p-values
                rgr_shap.nobs     = df_adj+int(rgr_shap.df_model)
                rgr_shap.df_resid = df_adj
            else: # bootstrap adjustemnt
                try:
                    m_adj = np.round(len(data)**(2.*ml_conv_rate)).astype(int)
                    if is_TS==True:
                        l_adj = len(data)-m_adj+1
                    else:
                        l_adj = n_boot
                    # init
                    df_mdl = len(rgr_shap.params)
                    b_temp, p_temp, t_temp, p_temp_1, d_temp_1 = np.zeros((l_adj,df_mdl)), np.zeros((l_adj,df_mdl)),\
                            np.zeros((l_adj,df_mdl)), np.zeros((l_adj,df_mdl)), np.zeros((l_adj,df_mdl))
                    p_temp_1_full = np.zeros(l_adj) # p values for contraint models
                    for i in range(l_adj):
                        if is_TS==True:
                            i_select = np.arange(i,i+m_adj) # rolling window
                        else:
                            i_select = np.random.randint(0,len(data),m_adj) # random sample
                        # Shapley regression
                        boot_rgr_shap         = smf.ols(formula=fml, data=decomp.iloc[i_select]).fit(cov_type=se_type,
                                                                                                     cov_kwds=cov_dict,use_t=True)
                        boot_rgr_shap.pvalues = rgr_shap.pvalues/2 # for one-sided test
                        # fill in results
                        b_temp[i,:] = boot_rgr_shap.params
                        p_temp[i,:] = boot_rgr_shap.pvalues
                        t_temp[i,:] = boot_rgr_shap.tvalues
                        # test contrained models (component robustness & model bias)
                        q = np.append(np.ones(n), n) # contraint values
                        #boot_rgr_shap.params = np.ones(len(boot_rgr_shap.params))
                        c_ttest = boot_rgr_shap.t_test((R,q),use_t=True) # t-test on constraints
                        p_temp_1[i,:]    = np.append(np.nan,c_ttest.pvalue[:-1]) # p-values
                        d_temp_1[i,:]    = np.append(np.nan,np.diff(c_ttest.conf_int(bias_conf_lvl),1)[:-1,0])
                        p_temp_1_full[i] = c_ttest.pvalue[-1]
                    # average values
                    rgr_shap.params  = b_temp.mean(0)
                    rgr_shap.pvalues = p_temp.mean(0)
                    rgr_shap.tvalues = t_temp.mean(0)
                except ZeroDivisionError:
                    print('ML adjustment would require negative sample size. Try changing test dof instead.')
                
    # extract results
    # ---------------
    df_rgr = pd.DataFrame({'shap_sign' : np.sign(rgr_ols.params), 'coefficient' : rgr_shap.params,
                           'p_zero'    : rgr_shap.pvalues,        'is_valid'    : rgr_shap.params>0,
                           't_zero'    : rgr_shap.tvalues})
    
    # test contrained models (component robustness & model bias)
    if mdl_type in ['Reg','ENet']: # linear models has beta^S=1 by definition
        df_rgr['p_one'] = np.append(np.nan,np.ones(n))
        df_rgr['w_one_'+str(int(100*bias_conf_lvl))] = np.append(np.nan,np.zeros(n))
    else:
        if boot_adj==False:
            if adj_for_ml_rate==True:
                cov_p = rgr_shap.normalized_cov_params*(df_raw/df_adj) # adjust dof
            else:
                cov_p = rgr_shap.normalized_cov_params
            c_ttest = rgr_shap.t_test((R,q),cov_p=cov_p,use_t=True) # t-test on constraints
            df_rgr['p_one'] = np.append(np.nan,c_ttest.pvalue[:-1]) # p-values
            # confidence interval with for non-rejection
            df_rgr[CI] = np.append(np.nan,np.diff(c_ttest.conf_int(bias_conf_lvl),1)[:-1,0])
        else:
            df_rgr['p_one'] = p_temp_1.mean(0)
            df_rgr[CI]      = d_temp_1.mean(0)

    # Shapley value share
    df_rgr['shap_share'] = ave_col_share(decomp[features],features,ave_m=True)
    
    # estimated SE of Shapley share
    df_rgr['shap_se'] = shapley_coeff_se(decomp[features],features,is_TS=is_TS)
    
    # sort by share
    if sort_share==True:
        df_rgr = df_rgr.sort_values('shap_share',ascending=False)
    # output
    all_cols = ['shap_sign','shap_share','shap_se','coefficient',
                'is_valid','p_zero','p_one',CI,'t_zero']
    df_rgr = df_rgr[all_cols]
    
    # regain original feature names
    df_rgr.index = [col_dict[i] for i in df_rgr.index]
    
    # print to screen
    if verbose==True:
        print(df_rgr.round(2).to_string())

    return df_rgr


def shapley_coeff_se(decomp,features,is_TS=False):
    '''Standard error estimation of Shapley coefficients.
    
    Inputs:
    -------
    
        decomp : pandas dataframe
            Shapley values decomposition of model outputs of dimension (nbr observations, nbr features)
            
        features : array-like
            model independent variables
            
        is_TS : Boolean (Default values = False)
            Whether to treat data as time series or not by accounting for feature autocorrelations
            
    Outputs:
        
        pandas series : indexed by features and values by Shapley coefficient SE. 
    
    '''
    
    features_shares = ave_col_share(decomp,features,ave_m=False)
    if is_TS==True:
        se = features_shares[features].apply(ts_se,axis=0)
    else:
        se = features_shares[features].sem()
        
    return se


def acov_delta(x,delta=1):
    '''Autocovariance of x with x shifted by delta.
    
    Inputs:
    -------
    
        x : array-like
            input time series
            
        delta : int (Default value = 1)
            shift in units of x indices
            
    Outputs:
    --------
    
        float : autocovaiance measure
    
    '''
    n, xs  = len(x), np.average(x)
    aCov, times = 0., 0.
    for i in np.arange(0, n-delta):
        aCov  += (x[i+delta]-xs)*(x[i]-xs)
        times +=1
    return aCov/times


def ts_se(x,kmax=-1):
    '''Standard error of time series mean.
    
    Inputs:
    -------
    
        x : array-like
            input time series
            
        kmax : int (Default value = -1, corresponds to int(np.round(np.sqrt(len(x)))))
            maximal order of autocovariance to be considered (kmax=0 chooses)
            
    Outputs:
    --------
    
        float : standard error measure
    
    '''
    
    n = len(x)
    if kmax==-1:
        kmax = int(np.round(np.sqrt(n)))
    
    acov_1plus = [((n-k)/float(n))*acov_delta(x,k) for k in range(1,kmax+1)]
    se         = np.sqrt(np.abs(acov_delta(x,0)+2*np.sum(acov_1plus))/n)
    
    return se



def ave_col_share(data,columns,ave_m=True):
    '''Average share of each column across rows.
    
    Inputs:
    -------
    
        data : pandas daaframe
            data for which to calculate share of columns
            
        columns : array-like
            names among which to calculate shares
            
        ave_m : Boolean (Default value = True)
            Whether to average shares for all observations
            
    Outputs:
    --------
    
        pandas dataframe or series 
            average column fraction averaged across rows or not
        
    '''

    data          = data[columns]
    column_shares = np.abs(data)/np.tile(np.abs(data).sum(1).values,(data.shape[1],1)).T
    # row average
    if ave_m==True:
        column_shares = column_shares.mean(0)
        
    return column_shares


def stack_shap_plot(df,target,features,mdl_type,min_share=-1,n_compos=-1,
                    colors=None,PoI={},country=None,ylim=None,save_fig=False,
                    save_name=None,unexpl=False,res=200):
    '''Stacked time series feature attribution plot.
    
    Inputs:
    -------
    
        df : pandas dataframe
            input data to be plotted
            
        features : array-like
            features for plot and labels
            
        mdl_type : str
            model type
            
        min_share : float in (0,1] (Default values = -1)
            Minimal explanatory Shapley value share to include a feature in plot
            
        n_compos : int, max len(features) (Default values = -1)
            number of leading features by Shapley values share to plot
            
        colors : dict (Default value = None)
            Colour palette for features
            
        PoI : dict (Default value = {})
            Dict of x coordinates to mark by vertical dashed line keyed by label and values by x-position
            
        country : str (Default value = None)
            Country label
            
        ylim : len-2 array (Default value = None)
            y-limits of plot
            
        save_fig L Boolean (Default value = False)
            If to save plot
            
        save_name : str (Default value = None)
            Name under which to save figure
            
        unexpl : Boolean (Default value = False)
            If to plot unxplained model component separately
            
        res : int (Default value = 200)
            Resolution in DPI for PNG figure if saved
            
    Outputs:
    --------
    
        none
    
    '''
    
    mdl_name_dict = {'NN':'NN','SVM':'SVM','Forest':'RF','Reg':'Reg','ENet':'EN'}
    if unexpl==True:
        features = list(features)+['unexplained']
    # only include leading components with > min_share
    if min_share>0:
        shap_share = (np.abs(df[features])/np.tile(np.abs(df[features]).sum(1).values,(df[features].shape[1],1)).T).mean()#.sort_values(ascending=False)
        incl, excl = [], []
        for name in features:
            if shap_share[name]>=min_share:
                incl.append(name)
            else:
                excl.append(name)
        if len(excl)>0:
            incl.append('others')
            features     = incl
            df['others'] = df[excl].sum(1)
    # only include leading components with > min_share
    if n_compos>0 and not min_share>0:
        shap_share = (np.abs(df[features])/np.tile(np.abs(df[features]).sum(1).values,(df[features].shape[1],1)).T).mean().sort_values(ascending=False)
        incl, excl, k = [], [], 0
        for name in shap_share.index:
            if k<n_compos:
                incl.append(name)
                k += 1
            else:
                excl.append(name)
        if len(excl)>0:
            incl.append('others')
            features     = incl
            df['others'] = df[excl].sum(1)
    # main plot
    plt.figure()
    if type(colors)==dict:
        if (min_share>0 or n_compos>0) and len(excl)>0:
            colors['others'] = '#c7c7c7'
        colors = [colors[f] for f in features]
    ax  = df[features].plot.bar(stacked=True,color=colors,width=1,alpha=0.75,figsize=(12,8)) # stackplot
    ax.grid(True, which='major', linestyle='-', linewidth=0.25,)
    ax.plot(df[target].values,         'k--',lw=1.5,label='{0} (target)'.format(target))
    if not unexpl==True:
        ax.plot(df[features].values.sum(1),'k-' ,lw=1.5, label='{0} (model)'.format(mdl_name_dict[mdl_type]))
    # insert points-of-interest
    if not len(PoI)==0:
        clr_dict = dict(zip(range(8),['b', 'g', 'r', 'c', 'm', 'y', 'k']))
        for i,point in enumerate(PoI):
            x_p = np.where(df.index==point)[0]
            ax.axvline(x_p,lw=1.5,c=clr_dict[i],ls='--',label=PoI[point])
    # ticks, labels & legends
    if not ylim==None:
        ax.set_ylim(ylim)
    x_ticks_i = range(0,len(df),40)
    x_ticks_v = [str(df.index[i])[:4] for i in x_ticks_i]
    plt.xticks(x_ticks_i,x_ticks_v,fontsize=12,rotation=0)
    plt.xlabel('year',fontsize=15)
    plt.ylabel('z-score',fontsize=15)
    plt.yticks(fontsize=13)
    # legend
    hnd, lab = ax.get_legend_handles_labels()
    ax.legend(hnd, lab, loc='upper center', bbox_to_anchor=(0.48, 1.11),
              prop={'size':11},ncol=math.ceil(len(lab)/2),framealpha=1)
    # country label
    if not country==None:
        plt.text(0.01,0.95,country+':'+target,fontsize=16,fontweight='bold',
                 alpha=0.6,family='monospace',transform=ax.transAxes)
    #save figure
    if save_fig==True:
        plt.savefig(save_name,dpi=res,bbox_inches="tight")
    plt.show()


def poly_fit_plot(x,y,c='k',lw=2,order=3,do_plot=True):
    '''Polynomial fit to x-y pairs of specified order.
    
    Inputs:
    -------
    
        x : array-like
            x positions
            
        y : array-like
            y positions
            
        c : str (Default value = 'k' [black])
            colour string
        
        lw : float (Default value = 2)
            line width of plot
            
        order : int (Default value = 3)
            order of fitted polynomial
            
        do_plot : Boolean (Default value = True)
            Whether to plot or return output
            
    Outputs:
    --------
    
    list (if do_plot==False): [x,y] of polynomial fit
        
    '''
    
    x2   = np.linspace(min(x),max(x)) # evenly spaced x for plotting
    pfit = poly.polyval(x2, poly.polyfit(x, y, order)) # fit
    if do_plot==True:
        plt.plot(x2, pfit, lw=lw, c=c, ls='--') # plot
    else:
        return [x2,pfit]
    

def print_screen_file(content,title=None,to_file=None,mode='a'):
    '''print results to screen and file.'''
    
    if not to_file==None:
        with open(to_file, mode) as f:
            f.write(title)  
            f.write(content)
        f.close()
        
    print(title)
    print(content)
    
    
def print_CV_results(cv_dict):
    '''Print train-test results from nested cross-validation.
    
    Inputs:
    -------
    
        cv_dict: dictionary containing results
        
    Outputs:
    --------
    
        none (print to screen)
    
    '''
    
    print('\nCV-Train-Test diagnostics ({0}):\n'.format(cv_dict['ID']))
    print('\t- MAE:  {0} (train),  {1} (test), {2} (GE),'.format(np.round(cv_dict['train_mae'],3),
          np.round(cv_dict['test_mae'],3),np.round(cv_dict['ma_ge'],3)))
    print('\t- RMSE: {0} (train),  {1} (test), {2} (GE),'.format(np.round(cv_dict['train_rmse'],3),
          np.round(cv_dict['test_rmse'],3),np.round(cv_dict['rms_ge'],3)))
    print('\t- MSE:  {0} (MSE), {1} (bias^2),  {2} (var).\n'.format(np.round(cv_dict['mse'],3), 
          np.round(cv_dict['bias2'],3),  np.round(cv_dict['var'],3)))



