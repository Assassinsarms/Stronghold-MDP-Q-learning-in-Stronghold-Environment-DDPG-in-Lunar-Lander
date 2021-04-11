# INM707_Coursework
 Reinforcement Learning Coursework

Task 1: 

In Task 1, a reinforcement learning environment known as Stronghold was created and modelled according to Markov Decision Process (MDP). The basic idea is that an agent is to infiltrate an enemy stronghold to gather intelligence. My inspiration for this environment was the video game ‘Hitman’ in which an assassin infiltrates various enemy territory to assassinate various targets. In this environment, the goal of the agent is less gruesome but the main idea is the same. 
# TO RUN, JUST IMPORT THE LIBRARIES AND THEN RUN ANY CELL 

Task 2:
In Task 2, Q-learning was implemented for the stronghold environment and multiple experiments were performed. There were 5 cases experimented: environment size variation, environment simplicity variation, learning rate variation, discount factor and epsilon variation. Each experiment had other hyper-parameters remain constant.
# TO RUN, JUST IMPORT THE LIBRARIES AND THEN RUN ANY CELL EXCEPT THE EPSILON DECAY EXAMPLE SINCE THIS IS COMMENTED OUT BY DEFAULT IN Q-LEARNING.PY 
# SOME OF THE OUTPUTS SHOW PLOTS AND SOME DO NOT TO SAVE TIME DURING COMPUTATION. THIS CAN BE TOGGLED BY THE VIEW VARIABLE IN EACH CELL 

Task 3:

In Task 3, a Deep Deterministic Policy Gradient algorithm was implemented for the Lunar-Lander OpenAI gym environment. This environment has a continuous action space therefore it was expected that DDPG would perform well since that is the kind of environment DDPG was designed for. There were 2 cases experimented: gamma variation and policy and Q-network learning rate variation. Each model was trained for 85 epochs.
# TO RUN A TEST, FIRST RUN THE SETUP CELLS AT THE BEGINNING OF THE NOTEBOOK AND THEN RUN THE TEST CELL AT THE BOTTOM OF THE NOTEBOOK
