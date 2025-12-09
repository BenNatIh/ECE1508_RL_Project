# ECE1508_RL_Project
Project:
Controlling Robotic Arm with RL

Baseline: 
To run the pid controller run pid_controller.py, To modify the params look into run_all_pid_baselines. You can modify seeds, noise levels, eval_fraction, max_episode_steps and eval episodes

PPO:
To train the PPO agent run main_ppo.py. Inside the file to change rewards from dense to sparse. change line 19 from TASK = 'reach-dense' to TASK = 'reach'.
Can also toggle using noisy obervations or HER by changing USE_NOISY_OBS and USE_HER respectively.

TD3:
To train and evaluate the TD3 agent, run 'td3_noise_sb3.py' for dense rewards and run 'td3_noise_sparse_sb3.py' for sparse rewards. To run these files, the installation of 'stable-baselines3' and 'rich' libraries are required. To adjust the observation noise level for 'td3_noise_sb3.py', change the 'OBS_NOISE_STD' variable in line 17. To adjust the observation noise level for 'td3_noise_sparse_sb3', change the 'noise_levels' list in line 72.

SAC:
To train and evaluate the SAC agent, run sac_agent.py.
This script automatically trains SAC across all seeds, reward types (dense and sparse), and observation noise levels.
All trained models, evaluation JSON files, and learning curve plots are saved under the results/ directory.
To modify noise levels, seeds, or reward settings, edit the parameters in the run_all_experiments() function at the bottom of the script.



