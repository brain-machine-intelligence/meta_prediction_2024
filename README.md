[![DOI](https://zenodo.org/badge/884656169.svg)](https://doi.org/10.5281/zenodo.14049443)


# meta_prediction_2024

Data, Simulation code and the Experiment code for paper

## 1. Simulation code 
- To perform the simulation, refer to the "SIMUL_cmd_send.m" file.
(main.py via SIMUL_cmd_send.m -> Data_analysis_new_opt.py via SIMUL_cmd_send.m -> Data_analysis_new_opt2.py -> shuffle_simulation_new_opt.py via SIMUL_cmd_send.m -> Shuffle_analysis_new_opt.py)
- Require Matlab (equal or higher than R2020a), python3 (equal or higher than 3.6.10)
- Expected output: .pkl or .mat files with model's behavior from the simulation codes, .mat and .png files with statistical summary from the analysis codes.
  - main.py : .pkl files with the single model's behavior during training
  - Data_analysis_new_opt.py : .pkl files with the single model's behavior after training
  - Data_analysis_new_opt2.py : .mat and .png files with statistical summary of the individual fitted models
  - shuffle_simulation_new_opt.py : .mat files with the model's behavior during shuffle test
  - Shuffle_analysis_new_opt.py) : .mat and .png files with statistical summary of models during shuffle tests
- Expected time: 2 hours per subjects
  - main.py : 16 hours per subjects
  - Data_analysis_new_opt.py : 30 minutes per subjects
  - Data_analysis_new_opt2.py : 30 minutes
  - shuffle_simulation_new_opt.py : 30 minutes per subjects
  - Shuffle_analysis_new_opt.py) : 30 minutes
    
## 2. Task code
- 2-stage MDP task code used in fMRI experiments in the paper. Run with "task_main_2020.m".
- Require Matlab (equal or higher than R2020a) and psychotoolbox package.
- Expected output : .m file with behavioral data.
- Expected time : 15~20 minutes for one task session.

## 3. Model fitting code
- Negative log likelihood was calculated for each action, using softmax function with action value (Q). Free parameters are optimized to minimize the sum of negative log likelihood (which means better estimation of human actions) using Nelder-Mead simplex algorithm. For more details, please refer to the supplementary document of Lee et al. (Neuron, 2014). Especially, "Parameter Estimation" section in supplementary methods explains all the details for inferring parameters.
The sum of negative log likelihood further used to calculate BIC and model comparison (RFX BMS).
- To perform the simulation, refer to the "job_swlee_arbitration_regressor_gen.m" file.
- Require Matlab (equal or higher than R2020a)
- Expected output : behavioral .m file with model fitting results (model parameters and BIC scores)
- Expected time : 2 hours per subejcts

## 4. Behavior Results
- Result_save : Raw behavior data from the human experiment
- SBJ_structure_sjh_con02_extended.mat : Model fitted with the raw behavior data in Result_save

## 5. Neural Results
- GLM : fMRI GLM results
- PSC : fMRI PSC results
- Regressors : Regressor files to do the GLM and PSC analysis.
- Require Matlab (equal or higher than R2020a) and SPM toolbox

# Acknowledgements
This work was funded by the the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT).(RS-2024-00341805), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2023-00233251, System3 reinforcement learning with high-level brain functions), the National Research Foundation of Korea(NRF) funded by the Korean government (MSIT) (No. RS-2024-00439903), Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT)(No.RS-2019-II190075 Artificial Intelligence Graduate School Program(KAIST), Electronics and Telecommunications Research Institute(ETRI) grant funded by the Korean government (24ZS1100, Core Technology Research for Self-Improving Integrated Artificial Intelligence System), and  Electronics and Telecommunications Research Institute(ETRI) grant funded by the Korean government [N01230878, Development of Beyond X-verse Core Technology for Hyper-realistic interactions by Synchronizing the Real World and Virtual Space].
