# ddpg agent in separate file (ddpgagent.py)
# trend functions moved to a separate file (trend.py)
# some more funcitons moved to separate file (RLUtilities.py)
# Yet to be done: generalizing data storage in csv files and trending function
# good for sharing
# has full networks

# import standard required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from itertools import product

# import user created classes and functions
from DDPGAgent import Buffer # # uses DDPGAgent.py stored in the same directory as this program
from DistColEnvDynInc_altrewards import DistColEnvDynInc # uses DistColEnvDynInc_altrewards.py stored in the same directory as this program
import trend  # uses trend.py stored in the same directory as this program
import RLUtilities # uses RLUtilities.py stored in the same directory as this program

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the distillation column environment 
env = DistColEnvDynInc()        
env.stdmult=0.002     # multiplicative noise for all except bottoms ipurity
env.stdadd=0.2      # additive noise for bottoms impurity
           
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high
lower_bound = env.action_space.low

# Set runtime and display information
online_run_time=1.25      # days
data_sampling_time=1    # minutes
RL_execution_time=6     # minutes
env.simtime=RL_execution_time
num_steps=int(online_run_time*24*60/RL_execution_time)
disphours=24             # hours to display when trending

# Set RL algorithm (DDPG) hyperparameters and create the agent
gamma = 0           # discounting factor
critic_lr = 0.06    # learning rate for critic network 
actor_lr = 0.1      # learning rate for actor network 
batch_size=64       # batch size for network trainig
buffer_capacity=batch_size*5    # buffer size for storage
tau = 0.005         # update rate for target networks
act_layer1=2
act_layer2=2
cir_slayer=16
cir_alayer=16
cir_layer2=32
cir_layer3=0
ddpgagent = Buffer(num_states, num_actions, upper_bound,lower_bound,buffer_capacity, batch_size, \
                   act_layer1,act_layer2,cir_slayer,cir_alayer,cir_layer2,cir_layer3,actor_lr,critic_lr,gamma,tau)   

# Set RL implementation tuning parameters
learn_control=0                 # 0=full learn with randon moves, 1=full control with agent moves
startcontrol=int(batch_size*3)  # number of steps after which learn_control is set to 1 (full control)
startlearn=int(batch_size)    #/2 number of steps before RL agent Critic network starts learning. 
startactortrain=int(batch_size*2) # number of steps before RL agent Actor network starts learning. 
replace_unsuitable_data=True    # Option to replace unsuitable data (due to state transition) by suitbale data
MoveSuppress=0.25*np.ones(num_actions)  # move suppression for each action

#**** the following is for tracking and trending purpose ****
Feed_sequence=np.zeros(num_steps+1)*math.nan
L_sequence=np.zeros((num_steps+1))*math.nan
V_sequence=np.zeros((num_steps+1))*math.nan
DP_sequence=np.zeros((num_steps+1))*math.nan
BI_sequence=np.zeros((num_steps+1))*math.nan
DPP_sequence=np.zeros((num_steps+1))*math.nan
BIP_sequence=np.zeros((num_steps+1))*math.nan
DP_targ_sequence=np.zeros((num_steps+1))*math.nan
BI_targ_sequence=np.zeros((num_steps+1))*math.nan
D_sequence=np.zeros((num_steps+1))*math.nan
B_sequence=np.zeros((num_steps+1))*math.nan
reward_sequence=np.zeros((num_steps+1))*math.nan
profit_sequence=np.zeros((num_steps+1))*math.nan
time_sequence=np.zeros((num_steps+1))*math.nan
NumPoints_sequence=np.zeros((num_steps+1))*math.nan
NumConstPoints_sequence=np.zeros((num_steps+1))*math.nan
SignCnt_sequence=np.zeros((num_steps+1))*math.nan
actorloss_sequence=np.zeros((num_steps+1))*math.nan
criticloss_sequence=np.zeros((num_steps+1))*math.nan
la0_sequence=np.zeros((num_steps+1))*math.nan
la1_sequence=np.zeros((num_steps+1))*math.nan
la2_sequence=np.zeros((num_steps+1))*math.nan
zero_sequence=np.zeros((num_steps+1))   # used for showing zero trend line

# following is for checking the uniqueness of the agent actions  
pattern=list(product(range(1, 4), repeat = num_actions))
pattern_count=np.zeros(len(pattern))
ActionPattern=np.zeros(num_actions)

#**** the following is for storing data in csv file for analysis later ****
s_y=""
s_cv=""
for i in range(batch_size):
    s_y=s_y+"y_"+str(i)+","
    s_cv=s_cv+"cv_"+str(i)+","

f2=open("RL_data1.csv","w")
f2.write("time, Feed, L, V, FeedLL, FeedL1, FeedL2, FeedH2, FeedH1, FeedHL, LLL, LL1, LL2, LH2, LH1, LHL, VLL, VL1, VL2, VH2, VH1, VHL, \
         DP, DPPred, BI, BIPred, D, B, DPH2, DPH1, DPHL, BILL, BIL1, BIL2, DLL, DL1, DL2, DH2, DH1, DHL, BLL, BL1, BL2, BH2, BH1, BHL, \
         reward, FeedC, LC, VC, Profit, prevstate0,prevstate1, prevstate2, action0, action1, action2, RA0, RA1, RA2, transition,\
         small_action, suppression, PA0, PA1, PA2, RandA0, RandA1, RandA2, SignCnt, AL, CL, oscillation, x_Feed, NoConst, Ratio, \
         altreward,"+s_y+s_cv+"\n")

"""
time, prev_state[], action[], state[], rewardused, reward, altreward, profit, LV contribution[], TV contribution[], laction[], raction[], transition, 
small_action, suppression, oscillation , NoConst, CL, signCnt, AL, dataRatio, {MVLL, MVLL1, MVLL2, absMV, MVHL2, MVHL1, MVHL}[],
{LVLL, LVLL1, LVLL2, LV,LVP, LVHL2, LVHL1, LVHL}[],{TVLL, TV,TVP, TVHL}[], y[], cv[],


"""

# initialize variables and environment
stepindex=0    
NumDataPoints=0
NoConstPoints=0
episodic_reward = 0

prev_state = env.reset("fixed")     # "fixed":for fixed initial operating point, "random": for random initial operating point
tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
action = ddpgagent.policy(tf_prev_state)
action=RLUtilities.SetAllActionsToZero(action)

# creating a placeholder for random action
raction=ddpgagent.policy(tf_prev_state)
raction=RLUtilities.SetAllActionsToZero(raction)

# creating a placeholder for agent learned action for no constraint state
laction=ddpgagent.policy(tf_prev_state)
laction=RLUtilities.SetAllActionsToZero(laction)

oscillation=False   # not used in the program

while True:
    print("Stepindex:",stepindex)
    
    # Start controlling after stepindex reaches startcontrol
    if stepindex>startcontrol:
        if stepindex==startcontrol+1:
            print("starting control")
        MoveSuppress=0.1*np.ones(num_actions)
        #MoveSuppress=0.2
        learn_control=1
    # reduce distillate purity target 40 steps after startcontrol    
    if stepindex>startcontrol+40:
        if stepindex==startcontrol+40+1:
            print("reducing distpurity target")
        env.targ_distpurity=88
    # increase distillate purity target 70 steps after startcontrol    
    if stepindex>startcontrol+70:
        if stepindex==startcontrol+70+1:
            print("increasing distpurity target")
        env.targ_distpurity=94

        
    prev_action=action
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
    action = ddpgagent.policy(tf_prev_state) # calculate agent action
    # save agent action before action adjustments. These are agent learned actions
    laction=RLUtilities.CopyAction(action, laction)
    # apply action restrictions and eliminate small actions
    action=RLUtilities.ActionRestriction(action, env.RestrictAction)
    action=RLUtilities.EliminateSmallAction(action, env.ahilimit, 0.05)
    
    # check for random action uniqueness after restriction are applied. 
    # If not unique, try again unitl a unique random action is found or until action_check_count_max number of attempts are done 
    temp_ra=np.zeros(num_actions)
    unique_action=False
    action_check_count_max=5        # number of attempts to find unique random action
    action_check_count=0
    while not unique_action and action_check_count<action_check_count_max:
        action_check_count+=1
        # generate random actions
        raction=RLUtilities.GenerateRandomAction(raction, env.ahilimit)
        # apply action restrictions
        raction=RLUtilities.ActionRestriction(raction, env.RestrictAction)
       
        unique_action=RLUtilities.UniqueActionCheck(raction, pattern, pattern_count, NumDataPoints>startlearn)
        
    # calculate combined action
    action=RLUtilities.CombineAction(action, raction, learn_control)

    # force two initial actions to be zero for better trending of variables        
    if stepindex<2:
        action=RLUtilities.SetAllActionsToZero(action)
    
    # Suppress combined action if suppression requirment is set
    if env.suppressaction:
        action=RLUtilities.ActionSuppression(action, MoveSuppress)
    
    # Enforce action if enforcement requirement is set
    action=RLUtilities.ActionEnforcement(prev_state, action, env.ahilimit)
    
    # Clamp MVs if implementing actions violates the limits
    action=RLUtilities.ActionClamp(action, env.aVal, env.aLL, env.aHL, env.ahilimit)
    
    # prepare action for use with the environment
    EnvAction=[action[0][0],action[0][1],action[0][2]]
    
    # implement action
    state, reward, done, info = env.step(EnvAction)

    # set reward to zero if the action is too small
    small_action=RLUtilities.SmallActionCheck(action, env.ahilimit, 0.05)
    if small_action: reward=0

    # check if there was a state transition due to action implementation
    transition=RLUtilities.TransitionCheck(prev_state, state)

    # Check if any constraint is active
    NoConst=RLUtilities.NoConstraintCheck(prev_state, 3)

    # train the RL agent if learn_control<1 else don't train
    if learn_control<1:
        # Save states, actions and reward for training if there is no state transition, no constraint is active and actions are not too small
        if not transition and not small_action and NoConst: 
            ddpgagent.record((prev_state, EnvAction, reward, state)) # save s(t-1), a(t), r(t) and s(t)
            NumDataPoints+=1
            #print("Step#:",stepindex,"#DataPoints:",NumDataPoints, " %useful data:",int(100*NumDataPoints/(stepindex+1)))
            reward_used=reward
            # set trainactor=True when sufficient data points are stored (for delayed start of actor training)
            if NumDataPoints>startactortrain: 
                trainactor=True
                if NumDataPoints==startactortrain+1: print("starting actor training")
            else: trainactor=False
            # start training critic (and actor as applicable) when sufficient data points are stored.
            if NumDataPoints>startlearn:
                if NumDataPoints==startlearn+1: print("starting critic training")
                ddpgagent.learn(trainactor)    
                # update the target networks
               
            # update the action pattern count
            pattern_count=RLUtilities.UpdatePatternCount(action, pattern, pattern_count)
               
            if NoConst:
                NoConstPoints+=1

        # Save alternate state and reward if there is state transition
        # the gym environment provides the alternate reward (env.altreward)
        elif not small_action and transition and replace_unsuitable_data:
            NoConstState=np.array([0,0,0])
            ddpgagent.record((NoConstState, EnvAction, env.altreward, NoConstState)) # save alternate s(t-1), a(t), r(t) and s(t)
            NumDataPoints+=1
            print("alternate data used")
            #print("Step#:",stepindex,"#DataPoints:",NumDataPoints, " %useful data:",int(100*NumDataPoints/(stepindex+1))," with Alternate data")
            reward_used=env.altreward
            # set trainactor=True when sufficient data points are stored (for delayed start of actor training)
            if NumDataPoints>startactortrain:
                trainactor=True
            else:
                trainactor=False
            # start training critic (and actor as applicable) when sufficient data points are stored.
            if NumDataPoints>startlearn:
                ddpgagent.learn(trainactor)
                # update the target networks

    episodic_reward += reward_used
    
    # the following for tracking the learning process 
    Feed_sequence[stepindex]=env.Feedval
    L_sequence[stepindex]=env.Lval
    V_sequence[stepindex]=env.Vval
    DP_sequence[stepindex]=env.distpurity
    BI_sequence[stepindex]=env.btmimpurity
    DPP_sequence[stepindex]=env.distpurityPred
    BIP_sequence[stepindex]=env.btmimpurityPred
    DP_targ_sequence[stepindex]=env.targ_distpurity
    BI_targ_sequence[stepindex]=env.targ_btmimpurity
    B_sequence[stepindex]=env.Bval
    D_sequence[stepindex]=env.Dval
    profit_sequence[stepindex]=env.profit/env.ProfitFactor
    reward_sequence[stepindex]=reward_used
    time_sequence[stepindex]=stepindex*env.simtime
    NumPoints_sequence[stepindex]=NumDataPoints
    NumConstPoints_sequence[stepindex]=NoConstPoints
    SignCnt_sequence[stepindex]=ddpgagent.SignCnt
    actorloss_sequence[stepindex]=ddpgagent.actorloss
    criticloss_sequence[stepindex]=ddpgagent.criticloss
    
    # saving learned action if the state represents that with no constriants
    if NoConst:
        la0_sequence[stepindex]=laction[0][0]
        la1_sequence[stepindex]=laction[0][1]
        la2_sequence[stepindex]=laction[0][2]
        
    
    # End episode when `done` is True
    if done or stepindex>=num_steps:
        break

    # store data in csv file for analysis later 
    s_y=""
    s_cv=""
    if NumDataPoints>startlearn:
        for i in range(batch_size):
            s_y=s_y+str(ddpgagent.y[i].numpy())+","
            s_cv=s_cv+str(ddpgagent.critic_value[i].numpy())+","
    s2=str(stepindex*env.simtime/60)+","+str(env.Feedval)+","+str(env.Lval)+","+str(env.Vval)+","+\
    str(env.Feed_LL)+","+str(env.Feed_LL+env.FeedGap)+","+str(env.Feed_LL+2*env.FeedGap)+","+str(env.Feed_HL-2*env.FeedGap)+","+str(env.Feed_HL-env.FeedGap)+","+str(env.Feed_HL)+","+\
    str(env.L_LL)+","+str(env.L_LL+env.LGap)+","+str(env.L_LL+2*env.LGap)+","+str(env.L_HL-2*env.LGap)+","+str(env.L_HL-env.LGap)+","+str(env.L_HL)+","+\
    str(env.V_LL)+","+str(env.V_LL+env.VGap)+","+str(env.V_LL+2*env.VGap)+","+str(env.V_HL-2*env.VGap)+","+str(env.V_HL-env.VGap)+","+str(env.V_HL)+","+\
    str(env.distpurity)+","+str(env.distpurityPred)+","+str(env.btmimpurity)+","+str(env.btmimpurityPred)+","+str(env.Dval)+","+str(env.Bval)+","+\
    str(env.targ_distpurity)+","+str(env.targ_distpurity+env.DPGap)+","+str(env.targ_distpurity+2*env.DPGap)+","+str(env.targ_btmimpurity-2*env.BIGap)+","+str(env.targ_btmimpurity-env.BIGap)+","+str(env.targ_btmimpurity)+","+\
    str(env.D_LL)+","+str(env.D_LL+env.DGap)+","+str(env.D_LL+2*env.DGap)+","+str(env.D_HL-2*env.DGap)+","+str(env.D_HL-env.DGap)+","+str(env.D_HL)+","+\
    str(env.B_LL)+","+str(env.B_LL+env.BGap)+","+str(env.B_LL+2*env.BGap)+","+str(env.B_HL-2*env.BGap)+","+str(env.B_HL-env.BGap)+","+str(env.B_HL)+","+\
    str(reward)+","+str(env.FeedConst)+","+str(env.LConst)+","+str(env.VConst)+","+str(env.profit)+","+\
    str(prev_state[0])+","+str(prev_state[1])+","+str(prev_state[2])+","+str(EnvAction[0])+","+str(EnvAction[1])+","+str(EnvAction[2])+","+\
    str(env.RestrictAction[0])+","+str(env.RestrictAction[1])+","+str(env.RestrictAction[2])+","+str(transition)+","+str(small_action)+","+str(env.suppressaction)+","+\
    str(laction[0][0])+","+str(laction[0][1])+","+str(laction[0][2])+","+str(raction[0][0])+","+str(raction[0][1])+","+str(raction[0][2])+","+\
    str(ddpgagent.SignCnt)+","+str(actorloss_sequence[stepindex])+","+str(criticloss_sequence[stepindex])+","+str(oscillation)+","+str(env.x_Feed[0])+","+\
    str(NoConst)+","+str(int(100*NumDataPoints/(stepindex+1)))+","+str(env.altreward)+","+s_y+s_cv+"\n"
    f2.write(s2)

    # the following is for trending information while training is going on
    # it opens a separate window and shows the trends for latest displayhours
    plt.clf()
    trend.actiontrend(4,3,1, "Feed",stepindex, time_sequence, Feed_sequence, env.Feed_LL, env.Feed_HL, env.FeedGap,disphours,env.simtime)
    trend.actiontrend(4,3,2, "L",stepindex, time_sequence, L_sequence, env.L_LL, env.L_HL, env.LGap,disphours,env.simtime)
    trend.actiontrend(4,3,3, "V",stepindex, time_sequence, V_sequence, env.V_LL, env.V_HL, env.VGap,disphours,env.simtime)
    trend.LVtrend(4,3,4, "D&B",stepindex, time_sequence, D_sequence, B_sequence, env.B_LL, env.B_HL, env.BGap,disphours,env.simtime)
    trend.LVLtrend(4,3,5, "Dist Purity",stepindex, time_sequence, DP_sequence, DPP_sequence,DP_targ_sequence, 75,105, env.DPGap,disphours,env.simtime)
    trend.LVHtrend(4,3,6, "btms impurity",stepindex, time_sequence, BI_sequence, BIP_sequence,env.targ_btmimpurity, -5,25, env.BIGap,disphours,env.simtime)
    trend.trend3(4,3,7, "CL & AL",stepindex, time_sequence, criticloss_sequence, actorloss_sequence,zero_sequence,disphours,env.simtime)    
    plt.xlabel("time - hours") 
    
    ax=plt.subplot(4,3,8)
    ax.grid()
    # Scatter plot of sampled rewards from the buffer against calculated rewards from Critic network 
    plt.scatter(ddpgagent.y,ddpgagent.critic_value)
    plt.xlabel("y")
    plt.ylabel("cv")
    plt.ylim(-0.75,0.75)
    plt.xlim(-0.75,0.75)
    trend.trend3(4,3,9, "LearnedActions",stepindex, time_sequence, la0_sequence, la1_sequence,la2_sequence,disphours,env.simtime,-0.125,0.125)    
    plt.xlabel("time - hours") 

    trend.trend3(4,3,10, "SignCnt",stepindex, time_sequence, SignCnt_sequence, zero_sequence,zero_sequence,disphours,env.simtime,0,batch_size)    
    #plt.plot(SignCnt_sequence)
    #plt.ylabel("Signcount")


    ax=plt.subplot(4,3,11)
    ax.bar(range(0,len(pattern)),pattern_count)
    plt.ylabel("pattern count")

    trend.trend3(4,3,12, "Profit",stepindex, time_sequence, profit_sequence, profit_sequence,profit_sequence,disphours,env.simtime,0,0.25)    

 
    plt.draw()
    plt.pause(0.02)        

    # prepare for next step
    prev_state = state
    stepindex+=1 

# episodic reward is calcualted as average rewards per step rather than accumulated reward.
episodic_reward=episodic_reward/stepindex 

# close the csv file
f2.close()

# execute if you want to see the model after training is complete
"""
print("actor model after training")
print(ddpgagent.actor_model.summary())
print(ddpgagent.actor_model.get_weights())
print("critic model after training")
print(ddpgagent.critic_model.summary())
print(ddpgagent.critic_model.get_weights())
"""
# execute this if we want to save the trained model 
"""
ddpgagent.actor_model.save_weights("DC_actor.h5")
ddpgagent.critic_model.save_weights("DC_critic.h5")
ddpgagent.target_actor.save_weights("DC_target_actor.h5")
ddpgagent.target_critic.save_weights("DC_target_critic.h5")
"""    
