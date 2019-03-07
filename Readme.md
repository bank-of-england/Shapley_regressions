Shapley regressions code base (BoE SWP 784) 
-------------------------------------------

This repository provides the code, data and results used for Bank of England Staff Working Paper 784

**"Shapley regressions: A framework for statistical inference on machine learning models"**

by Andreas Joseph (March 2019). 

The paper introduces a well-motivated and rigorous approach to address the black-box critique of machine learning models. Model interpretability is reduced to a multiple linear regression analysis - one of the most transparent and most widely used modelling techniques.

The output of machine learning models can now be presented as a regression table. The example below shows inference results for modelling UK and US unemployment using quarterly macroeconomic time series and comparing several machine learning models (columns 1-3 for each country) with a linear regression (Reg column). Please see Table 4 in the paper for details. 

![](https://github.com/bank-of-england/Shapley_regressions/blob/master/figures/U_Shap_reg_table.png)

The material provided here allows to reproduce all empirical and simulation results. It is not intended as a stand-alone package. However, parts of it may be transfered to other applications. No warranty is given. Please consult the licence file. 

Should you have any queries or spot an issue, please email to andreas.joseph@bankofengland.co.uk or
raise an Issue within the repository.

**Link to paper:** www.bankofengland.co.uk/working-paper/2019/shapley-regressions-a-framework-for-statistical-inference-on-machine-learning-models

**Download of full results:** https://www.dropbox.com/s/bkdjpbqrabgtwr4/SWP784_all_results.zip?dl=0



Code structure
--------------

	- 1_macro_Shapley_regressions.py: UK and US macroeconomic time series analysis using 
		machine learning (ML) models and Shapley regressions for statistical inference (section 5.2 of paper).
	- 2a_ML_inference_simulation.py: Simulation of polynomial data-generating processes and
		ML inference based on Shapley decompositions and reconstruction
		(suited for parallel/cloud processing, section 5.1 of paper).
	- 2b_ML_inference_analysis.py: Collection of simulation results and graphical output (section 5.1 of paper). 
	- ML_inference_aux.py: Auxiliary code for parts 1 and 2, application-specific inputs and 
		general functions (partly inherited from https://github.com/andi-jo/ML_projection_toolbox).
		shapley_coeffs() calculates Shapley share coefficients (SSC).


Instructions
------------

	- Parts 1 and 2 are independent from each other.
	- Part 2b depends on 2a or on pre-computed results (SWP results are provided in
		ML_inf_joint_results_swp.pkl).
	- The "main_dir" variable needs to be set in both parts.
	- options can be set at the beginning of parts 1 and 2 (a and b).
	- Please consult the comments in the codes and docstrings for further documentation.


Dependencies & versions
-----------------------

	- python (3.6.8, Anaconda distribution has been used)
	- numpy (1.15.4)
	- scipy (1.2.0)
	- pandas (0.24.1)
	- sklearn (0.20.2)
	- shap (0.28.3)
	- statsmodels (0.9.0)
	- matplotlib (3.0.2)
	- patsy (0.5.1)


Data & sources
--------------

Data description:

	- Quarterly marcoeconomic time series (UK: 1955Q1-2017Q4, US: 1965Q1-2017Q4).
	- Series are either yoy percentage changes or 1st difference (see Table 2 of the paper).
	- For the analysis, series are standardised to have mean zero and standard deviation one.
	- raw data and standardised series provided.
	- series names: GDP, labour productivity, broad money, private non-financial sector debt, 
		unemployment rate, household gross-disposable income, consumer price inflation, 
		central banks main policy rate, current account balance, effective exchange rate.

Individual sources by ID:

	- BOE: IUQLBEDR, XUQLBK82, IUQLBEDR, LPQAUYN.
	- ONS: D7BT, UKEA,PGDP, PRDY, MGSX.
	- BIS: US private sector debt: Q:US:P:A:M:XDC:A
	       UK: ERI, GBP/USD (1955 only).
	- OECD: US CPI, US M3, US GDP, US Unemployment, US CA.
	- FRED: RNUSBIS, FEDFUNDS, PRS85006163, A229RX0:
	- A Millennium of UK Data, Ryland Thomas (2017): private sector debt, 
		M4, labour productivity.
