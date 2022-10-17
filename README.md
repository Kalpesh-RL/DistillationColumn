The code provided here is associated with a paper submitted for potential publication in Computers & Chemical Engineering Journal.

There are a total of 6 files as follows. 

requirements.txt : This file contains the list of all python libraries required to be able to run the code. You may use the following command to install the libraries in a virtual environment

    pip install -r requirements.txt

DistColRL.py : This is the main program file to be executed. Please copy this and the other 4 python files in the same directory.

DistColEnvDynInc_altrewards.py : This is a python class that codes an openAI gym environment to simulate a distillation column using Gekko python library. 

DDPGAgent.py : This is a python class that codes the RL agent using DDPG algorithm.

RLUtilities.py : This is a collection of functions associated with implementing RL that are called by DistColRL.py main program

trend.py : This is a collection of functions associated with trending variables to monitor the RL agent training and subsequent use of trained agent.
