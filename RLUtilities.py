import random
import numpy as np

# function to implement action suppression
def ActionSuppression(action, MoveSuppress):
    for i in range(len(MoveSuppress)):
        action[0][i]=MoveSuppress[i]*action[0][i]
    return action

# function to implement action restriction
def ActionRestriction(action, RestrictAction):
    for i in range(len(RestrictAction)):
        if (RestrictAction[i]==1 or RestrictAction[i]==2) and action[0][i]>0:
            action[0][i]=0      # don't allow increase
        if (RestrictAction[i]==-1 or RestrictAction[i]==2) and action[0][i]<0:
            action[0][i]=0      # don't allow decrease
    return action

# function to implement action enforcement
def ActionEnforcement(prev_state, action, ahilimit):
    for i in range(len(action[0])):
        if prev_state[i]>0: action[0][i]=-ahilimit[i]/2
        if prev_state[i]<0: action[0][i]=ahilimit[i]/2
    return action

# function to ensure limits for absolute actions is honored by clamping the actions
def ActionClamp(action, aValue, aLL, aHL, ahilimit):
    for i in range(len(action[0])):
        if aValue[i]+action[0][i]<=aLL[i]:
            action[0][i]=min(aLL[i]-aValue[i],ahilimit[i])
        if aValue[i]+action[0][i]>=aHL[i]:
            action[0][i]=max(aHL[i]-aValue[i],-ahilimit[i])
    return action

# function to generate random actions for exploration   
def GenerateRandomAction(raction, ahilimit):
    for i in range(len(raction[0])):
        raction[0][i]=random.randint(-1,1)*ahilimit[i]
    return raction

# function to combine random and agent action to calculate action to be implemented
def CombineAction(action, raction, learn_control):
    for i in range(len(action[0])):
        action[0][i]=raction[0][i]*(1-learn_control) + action[0][i]*learn_control
    return action

# function to set all actions to zero
def SetAllActionsToZero(action):    
    for i in range(len(action[0])):
        action[0][i]=0
    return action

# function to copy actions
def CopyAction(action, laction):
    for i in range(len(action[0])):
        laction[0][i]=action[0][i]
    return laction

# function to check uniqueness of actions with diferent threshold counts for before and after training time frames
def UniqueActionCheck(raction, pattern, pattern_count, training_started):
    unique_action=False
    ActionPattern=np.zeros(len(raction[0]))
    for k in range(len(raction[0])):
        if raction[0][k]>0:   ActionPattern[k]=1
        elif raction[0][k]<0: ActionPattern[k]=3
        else:                 ActionPattern[k]=2

    if not training_started: count_threshold=2      # count threshold before agent training starts
    else: count_threshold=4                         # count threshold after agent training starts
            
    for j in range(len(pattern)):
        if (np.array_equal(pattern[j],ActionPattern) and pattern_count[j]<count_threshold):
            unique_action=True
    return unique_action

# function to compare actions against all the possible action patterns and update count for each action pattern 
def UpdatePatternCount(action, pattern, pattern_count):
    ActionPattern=np.zeros(len(action[0]))
    for k in range(len(action[0])):
        if action[0][k]>0:   ActionPattern[k]=1
        elif action[0][k]<0: ActionPattern[k]=3
        else:              ActionPattern[k]=2
        #print("envaction,action pattern:", EnvAction,ActionPattern)
        
    for j in range(len(pattern)): 
        if np.array_equal(pattern[j],ActionPattern): 
            #print("true for",j)
            pattern_count[j]+=1
    return pattern_count

# fuction to check for small actions    
def SmallActionCheck(action, ahilimit, factor=0.05):
    temp=0
    for i in range(len(action[0])):
        if abs(action[0][i])<ahilimit[i]*0.05 and abs(action[0][i])>0: temp=temp+1
    if temp==len(action[0]):
        small_action=True
    else:
        small_action=False
    return small_action

# fuction to eliminate small actions by setting them to zero
def EliminateSmallAction(action, ahilimit, factor=0.05):
    for i in range(len(ahilimit)):
        if abs(action[0][i])<ahilimit[i]*factor:
            action[0][i]=0
    return action

# fuction to check is there is a state transition (variable crossing a limit or target )           
def TransitionCheck(prev_state, state):
    temp=0
    for i in range(len(state)):
        if prev_state[i]!=state[i]: temp=temp+1
    if temp>0: transition=True 
    else: transition=False
    return transition

# fuction to check if there are no constraints active
def NoConstraintCheck(prev_state, numLV):
    temp=0
    for i in range(numLV):
        if prev_state[i]==0: temp=temp+1
    if temp==numLV: NoConst=True
    else: NoConst=False
    return NoConst


    
