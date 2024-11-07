[![DOI](https://zenodo.org/badge/884656169.svg)](https://doi.org/10.5281/zenodo.14049443)


# meta_prediction_2024

Data, Simulation code and the Experiment code for paper

## 1. Simulation code
- To perform the simulation, refer to the "SIMUL_cmd_send.m" file.

## 2. Task code
- 2-stage MDP task code used in fMRI experiments in the paper. Run with "task_main_2020.m".

## 3. Model fitting code
- Negative log likelihood was calculated for each action, using softmax function with action value (Q). Free parameters are optimized to minimize the sum of negative log likelihood (which means better estimation of human actions) using Nelder-Mead simplex algorithm. For more details, please refer to the supplementary document of Lee et al. (Neuron, 2014). Especially, "Parameter Estimation" section in supplementary methods explains all the details for inferring parameters.

The sum of negative log likelihood further used to calculate BIC and model comparison (RFX BMS).

## 4. Behavior Results
- Result_save : Raw behavior data from the human experiment
- SBJ_structure_sjh_con02_extended.mat : Model fitted with the raw behavior data in Result_save

## 5. Neural Results
- GLM : fMRI GLM results
- PSC : fMRI PSC results
- Regressors : Regressor files to do the GLM and PSC analysis.

# Acknowledgements
This work was funded by the the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT).(RS-2024-00341805), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2023-00233251, System3 reinforcement learning with high-level brain functions), the National Research Foundation of Korea(NRF) funded by the Korean government (MSIT) (No. RS-2024-00439903), Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT)(No.RS-2019-II190075 Artificial Intelligence Graduate School Program(KAIST), Electronics and Telecommunications Research Institute(ETRI) grant funded by the Korean government (24ZS1100, Core Technology Research for Self-Improving Integrated Artificial Intelligence System), and  Electronics and Telecommunications Research Institute(ETRI) grant funded by the Korean government [N01230878, Development of Beyond X-verse Core Technology for Hyper-realistic interactions by Synchronizing the Real World and Virtual Space].
