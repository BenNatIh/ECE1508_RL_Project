# ECE1508_RL_Project
Project:
Controlling Robotic Arm with RL

To train the PPO agent run main_ppo.py. Inside the file to change rewards from dense to sparse. change line 19 from TASK = 'reach-dense' to TASK = 'reach'.
Can also toggle using noisy obervations or HER by changing USE_NOISY_OBS and USE_HER respectively.
