Shapley regressions Code Base (BoE SWP 784) 
-------------------------------------------

This repository provides the full code base, data and results used for Bank of England Staff Working Paper 784

"Shapley regressions: A framework for statistical inference on machine learning models" 

by Andreas Joseph (March 2019). 

The material allows to reproduce all empirical and simulation results. 
It is not intended as a stand-alone package. However, parts of it may be transfered to other applications. 
No warranty is given. Please consult the licence file. 

Should you have any queries or spot an issue, please mail to andreas.joseph@bankofengland.co.uk or
raise an Issue within the repository.

Link to paper: TBD

Download of results: https://www.dropbox.com/s/bkdjpbqrabgtwr4/SWP784_all_results.zip?dl=0



Code structure:
---------------

	- 1_macro_Shapley_regressions.py: UK and US macroeconomic time series analysis using 
		machine learning (ML) models and Shapley regressions for statistical inference.
	- 2a_ML_inference_simulation.py: Simulation of polynomial data-generating processes and
		ML inference based on Shapley decompositions and reconstruction
		(suited for parallel/cloud processing).
	- 2b_ML_inference_analysis.py: Collection of simulation results and graphical output. 
	- ML_inference_aux.py: Auxiliary code for parts 1 and 2, application-specific inputs and 
		general functions (partly inherited from https://github.com/andi-jo/ML_projection_toolbox).
		shapley_coeffs() calculates Shapley share coefficients (SSC).


Instructions:
-------------

	- Parts 1 and 2 are independent from each other.
	- Part 2b depends on 2a or on pre-computed results (SWP results are provided in
		ML_inf_joint_results_swp.pkl).
	- The "main_dir" variable needs to be set in both parts.
	- options can be set at the beginning of parts 1 and 2 (a and b).
	- Please consult the comments in the codes and docstrings for further documentation.


Main packages dependencies and versions:
----------------------------------------

	- python (3.6.8, Anaconda distribution has been used)
	- numpy (1.15.4)
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
	- all series are standardised to mean zero and std one.
	- series names: GDP, labour productivity, broad money, private non-financial sector debt, 
		unemployment rate, household gross-disposable income, consumer price inflation, 
		central banks main policy rate, current account balance, effective exchange rate. 
    	- raw data and standardised series provided.

Individual sources by ID:

	- BOE: IUQLBEDR, XUQLBK82, IUQLBEDR, LPQAUYN.
	- ONS: D7BT, UKEA,PGDP, PRDY, MGSX.
	- BIS: US private sector debt: Q:US:P:A:M:XDC:A
	       UK: ERI, GBP/USD (1955 only).
	- OECD: US CPI, US M3, US GDP, US Unemployment, US CA.
	- FRED: RNUSBIS, FEDFUNDS, PRS85006163, A229RX0:
	- A Millennium of UK Data, Ryland Thomas (2017): private sector debt, 
		M4, labour productivity.
