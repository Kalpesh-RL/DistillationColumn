import gym
from gym import spaces
import numpy as np
import random
from gekko import GEKKO

class DistColEnvDynInc(gym.Env):
    """
    Description:
        This environment simulates a dynamic binary distillation column.
        It is based on code provided by Gekko at the following site
        http://apmonitor.com/wiki/index.php/Apps/DistillationColumn

    Assumptions:
        1. No tray holdup
        2. constant relative volatility
        3. Energy balance not included
        
    Observation:
        Type: Box(9)
        Num     Observation                 Min     Max
        0       Feed MV constraint          -100    100
        1       L(Reflux) MV constraint     -100    100
        2       V(Vapor) MV constraint      -100    100

    Action: (incremental)
        Type: Box(3)
        Num     Action                   Min     Max
        0       Feed                    -0.5       0.5
        1       Reflux (L)              -0.1       0.1
        2       Vapour (V)              -0.12      0.12

        The mole fraction of the lighter component in the feed (x_feed) can be changed to introduce disturbance.
        
    Reward: (incremental as the actions are incremental)
        Reward is a continuous variable that represents the impact of an action
        It incorporates the following
        1: contribution due to constraints
        2: contribution due to efficiency or profit
        An alternate reward is also calculated which excludes the contribution of constraints.
        This represents the alternate reward associated with alternate limits as mentioned in Section 4.4 in the paper.
        
    Starting State:
        This is set by calling reset() function 
        It can be assigned either randomly or to a fixed predefined state hardcoded here

    Episode Termination:
        When maximum episode length is reached or 
        Distillate or bottoms flow is zero
        Distillate purity > 1
        Bottoms purity < 0

    General information:        
        The comments in the code mentions "the paper" which refers to the paper submitted to CPC. 
        Though this code is associated with that paper it can be indepenently used too.
    """    
    
    def __init__(self):
        # defining observation and action space
        shilimit = np.array([100,100,100],dtype=np.float32)
        slolimit = np.array([-100,-100,-100],dtype=np.float32)
        self.observation_space = spaces.Box(slolimit, shilimit, dtype=np.float32)
        self.state=np.array([0,0,0])

        #Setting high and low limits and max change for actions
        self.Feed_HL=2.3
        self.Feed_LL=1.75
        self.L_HL=6
        self.L_LL=4 #2
        self.V_HL=7
        self.V_LL=5#2
        self.x_Feed_HL=0.6
        self.x_Feed_LL=0.4
        self.aHL=np.array([self.Feed_HL,self.L_HL,self.V_HL],dtype=np.float32)
        self.aLL=np.array([self.Feed_LL,self.L_LL,self.V_LL],dtype=np.float32)
        
        self.Feed_Chg=0.05
        self.L_Chg=0.1
        self.V_Chg=0.12
       
        self.ahilimit = np.array([self.Feed_Chg,self.L_Chg,self.V_Chg],dtype=np.float32)
        self.alolimit = np.array([-self.Feed_Chg,-self.L_Chg,-self.V_Chg],dtype=np.float32)
        self.action_space = spaces.Box(self.alolimit, self.ahilimit, dtype=np.float32)
        
        # quality targets variables (TVs)
        self.targ_distpurity=92
        self.targ_btmimpurity=8
        self.DPGap = 3
        self.BIGap = 3
        # limits for constraint varibales (limit variables LVs)
        self.D_HL=2
        self.D_LL=0.4
        self.B_HL=2
        self.B_LL=0.4

        # Initialize gekko Model
        self.m = GEKKO(remote=False)
        self.numtray=30
        feedtray=16
        numstg=self.numtray+2
        self.simtime=6                  # simulation output given every 6 minutes
        self.simsteps=self.simtime+1    # 7 steps for 6 minutes simulation output means internally simulation will run every 1 minute

        # Define values for variables in the model that can be changed
        self.L=self.m.Param(value=5.0)
        self.V=self.m.Param(value=6.0)
        self.Feed=self.m.Param(value=2.0)
        self.x_Feed=self.m.Param(value=.5)
        
        # Define values for constants in the model
        #Relative volatility = (yA/xA)/(yB/xB) = KA/KB = alpha(A,B)
        self.vol=self.m.Const(value=1.6)
        # Total molar holdup on each tray
        self.atray=self.m.Const(value=.25)
        # Total molar holdup in condenser
        self.acond=self.m.Const(value=1.0)
        # Total molar holdup in reboiler
        self.areb=self.m.Const(value=.5)
        
        # mole fraction of component A
        self.x=[]
        for i in range(numstg):
            self.x.append(self.m.Var(.3))
        
        # Define intermediates
        self.D=self.m.Intermediate(self.V-self.L)
        self.FL=self.m.Intermediate(self.Feed+self.L)
        self.B=self.m.Intermediate(self.Feed-self.D)
        
        # vapor mole fraction of Component A
        # From the equilibrium assumption and mole balances
        # 1) vol = (yA/xA) / (yB/xB)
        # 2) xA + xB = 1
        # 3) yA + yB = 1
        self.y=[]
        for i in range(numstg):
            self.y.append(self.m.Intermediate(self.x[i]*self.vol/(1+(self.vol-1)*self.x[i])))
        
        # condenser
        self.m.Equation(self.acond*self.x[0].dt()==self.V*(self.y[1]-self.x[0]))
        
        # column stages above feed tray
        n=1
        for i in range(feedtray-1):
            self.m.Equation(self.atray * self.x[n].dt() ==self.L*(self.x[n-1]-self.x[n]) - self.V*(self.y[n]-self.y[n+1]))
            n=n+1
        # feed tray
        self.m.Equation(self.atray * self.x[feedtray].dt() == self.Feed*self.x_Feed + self.L*self.x[feedtray-1] - self.FL*self.x[feedtray] - self.V*(self.y[feedtray]-self.y[feedtray+1]))
        
        # column stages below feed tray
        n=feedtray+1
        for i in range(self.numtray-feedtray):
            self.m.Equation(self.atray * self.x[n].dt() == self.FL*(self.x[n-1]-self.x[n]) - self.V*(self.y[n]-self.y[n+1]))
            n=n+1
        # reboiler
        self.m.Equation(self.areb  * self.x[self.numtray+1].dt() == self.FL*self.x[self.numtray] - (self.Feed-self.D)*self.x[self.numtray+1] - self.V*self.y[self.numtray+1])

        # steady state solution
        self.m.solve(disp=False)
        
        self.distpurity=self.x[0][0]*100
        self.btmimpurity=self.x[self.numtray+1][0]*100
        self.prev_distpurity=self.distpurity
        self.prev_btmimpurity=self.btmimpurity
        self.distpurityPred=self.distpurity
        self.btmimpurityPred=self.btmimpurity
        
        self.roc_distpurity=0
        self.roc_btmimpurity=0
        self.pred_horizon=6           # in minutes : used for estimating steady state value      
        self.Feedval=self.Feed[0]
        self.Lval=self.L[0]
        self.Vval=self.V[0]
        self.Dval=self.D[0]
        self.Bval=self.B[0]
        self.L2D=self.Lval/self.Dval
        self.V2B=self.Vval/self.Bval
        self.aVal=np.array([self.Feedval,self.Lval,self.Vval],dtype=np.float32)

        # pricing for profit calc
        self.FeedPrice=1
        self.DPrice=1.2
        self.BPrice=1.1
        self.LPrice=0.02
        self.VPrice=0.02

        self.PrevValue=0        # previous value for incremental reward calculation
        self.NewValue=0         # new value for incremental reward calculation
        self.altPrevValue=0     # Prevoius value for alernate reward calculation
        self.altNewValue=0      # New value for alernate reward calculation
        self.ConstFactor=-5     # scaling factor for constriant contribution
        self.ProfitFactor=50    # scaling factor for efficiency(profit) contribution
        
        self.FeedGap = self.Feed_Chg
        self.LGap = self.L_Chg
        self.VGap = self.V_Chg
        self.BGap = 0.1
        self.DGap = 0.1
        
        self.state=self.StateCalc()
        self.reward, self.altreward=self.RewardCalc()
        self.reward=0
        self.altreward=0

        # noise parameters
        self.mean=1
        self.stdmult=0.002  #used to introduce multiplicative noise for distillate purity, Distillate and bottoms products
        self.stdadd=0.2     #used to introduce additive noise for bottoms impurity

    
    def reset(self, type="random"):
        self.m.options.imode=1
        done=(1==2)
        while not done:
            if type=="random":
                # If type=random then the changable variables in the model are set randomly
                self.Feed.value=random.uniform(self.Feed_LL-2*self.Feed_Chg,self.Feed_HL+2*self.Feed_Chg)
                self.x_Feed.value=random.uniform(self.x_Feed_LL,self.x_Feed_HL)
                self.L.value=random.uniform(self.L_LL-2*self.L_Chg,self.L_HL+2*self.L_Chg)
                self.V.value=random.uniform(self.V_LL-2*self.V_Chg,self.V_HL+2*self.V_Chg)
            else:
                # If type is anothing other than "random" then the changable variables are set to predefined values
                self.Feed.value=2
                self.x_Feed.value=0.5
                self.L.value=5.0
                self.V.value=6.0
            # steady state solution
            self.m.solve(disp=False)
            if self.D[0]>0.1 and self.B[0]>0.1 and self.x[0][0]<=1 and self.x[self.numtray+1][0]>=0:
                done=(1==1)
        
        self.distpurity=self.x[0][0]*100
        self.btmimpurity=self.x[self.numtray+1][0]*100
        self.Feedval=self.Feed[0]
        self.prev_distpurity=self.distpurity
        self.prev_btmimpurity=self.btmimpurity
        self.distpurityPred=self.distpurity
        self.btmimpurityPred=self.btmimpurity
        self.Lval=self.L[0]
        self.Vval=self.V[0]
        self.Dval=self.D[0]
        self.Bval=self.B[0]
        self.L2D=self.Lval/self.Dval
        self.V2B=self.Vval/self.Bval
        self.aVal=np.array([self.Feedval,self.Lval,self.Vval],dtype=np.float32)
        self.SuppressActionCalc()
        self.RestrictActionCalc()
        self.state=self.StateCalc()
        self.reward, self.altreward=self.RewardCalc()
        self.reward=0
        self.altreward=0
        return self.state

    def step(self, Action):
        self.m.options.imode=7

        self.Feed.value=self.Feed.value+Action[0]
        self.L.value=self.L.value+Action[1]
        self.V.value=self.V.value+Action[2]
        self.Feedval=self.Feed[0]
        self.Lval=self.L[0]
        self.Vval=self.V[0]
        self.aVal=np.array([self.Feedval,self.Lval,self.Vval],dtype=np.float32)

        self.m.time=np.linspace(0,self.simtime,self.simsteps)
        self.m.solve(disp=False)
        
        self.done=(1==1)
        if self.D[self.simsteps-1]>0.1 and self.B[self.simsteps-1]>0.1 and self.x[0][self.simsteps-1]<=1 and self.x[self.numtray+1][self.simsteps-1]>=0:
            self.done=(1==2)
        msg=""
        if self.done:
            if (self.D[self.simsteps-1]<0.1 ):
                msg=msg+", D<0.1 issue:"+str(int(self.D[self.simsteps-1]*100)/100)
            if (self.B[self.simsteps-1]<0.1 ):
                msg=msg+", B<0.1 issue:"+str(int(self.B[self.simsteps-1]*100)/100)
            if (self.x[0][self.simsteps-1]>1 ):
                msg=msg+", DP>1 issue:"+str(int(self.x[0][self.simsteps-1]*100)/100)
            if (self.x[self.numtray+1][self.simsteps-1]<0 ):
                msg=msg+", BI<0:"+str(int(self.x[self.numtray+1][self.simsteps-1]*100)/100)
            
        self.distpurity=self.x[0][self.simsteps-1]*100*np.random.normal(self.mean,self.stdmult)
        self.btmimpurity=self.x[self.numtray+1][self.simsteps-1]*100+np.random.normal(0,self.stdadd)
        self.prev_distpurity=self.x[0][self.simsteps-3]*100*np.random.normal(self.mean,self.stdmult)
        self.prev_btmimpurity=self.x[self.numtray+1][self.simsteps-3]*100+np.random.normal(0,self.stdadd)
        self.Dval=self.D[self.simsteps-1]*np.random.normal(self.mean,self.stdmult)
        self.Bval=self.B[self.simsteps-1]*np.random.normal(self.mean,self.stdmult)
        self.L2D=self.Lval/self.Dval
        self.V2B=self.Vval/self.Bval

        self.SuppressActionCalc()
        self.RestrictActionCalc()
        self.state=self.StateCalc()
        self.reward, self.altreward=self.RewardCalc()
        return self.state, self.reward, self.done, msg
        
    def render(self):
        pass
    
    def close(self):
        pass
    
    def StateCalc(self, GapMult=1):
        # For calculating the State of the environment, this function is called with GapMult=1 to consider the unsafe zones
        # RestrictActionCalc() calls this with GapMult=2 in order to consider safe zones
        # The safe and unsafe zones are as mentioned in the paper.
        # roc calcualtion
        self.roc_distpurity=(self.distpurity-self.prev_distpurity)/2  
        self.roc_btmimpurity=(self.btmimpurity-self.prev_btmimpurity)/2 
        # predicted value calculation
        self.distpurityPred=self.distpurity+self.roc_distpurity*self.pred_horizon
        self.btmimpurityPred=self.btmimpurity+self.roc_btmimpurity*self.pred_horizon

        self.FeedDevL=max(0,self.Feed_LL-self.Feedval+self.FeedGap*GapMult)
        self.FeedDevH=max(0,self.Feedval-self.Feed_HL+self.FeedGap*GapMult)
        self.LDevL=max(0,self.L_LL-self.Lval+self.LGap*GapMult)
        self.LDevH=max(0,self.Lval-self.L_HL+self.LGap*GapMult)
        self.VDevL=max(0,self.V_LL-self.Vval+self.VGap*GapMult)
        self.VDevH=max(0,self.Vval-self.V_HL+self.VGap*GapMult)
        
        self.DDevL=max(0,self.D_LL-self.Dval+self.DGap*GapMult)
        self.DDevH=max(0,self.Dval-self.D_HL+self.DGap*GapMult)
        self.BDevL=max(0,self.B_LL-self.Bval+self.BGap*GapMult)
        self.BDevH=max(0,self.Bval-self.B_HL+self.BGap*GapMult)
        self.DPDevL=max(0,self.targ_distpurity-self.distpurityPred+self.DPGap*GapMult)/100
        self.BIDevH=max(0,self.btmimpurityPred-self.targ_btmimpurity+self.BIGap*GapMult)/100

        # Adjustments for use when calculating RestrictActions. Will be used when tehis funciton is called wiht GapMult=2
        if self.DPDevL>0:
            BIDevHAdj=0
        else:
            BIDevHAdj=self.BIDevH
        if self.BIDevH>0:
            DPDevLAdj=0
        else:
            DPDevLAdj=self.DPDevL
            
        self.FeedAtLowConst=max(self.FeedDevL,self.BDevL,self.DDevL)   #Feed low restricted for material balance only not for quality
        self.FeedAtHighConst=max(self.FeedDevH,self.BDevH,self.DDevH, min(self.VDevH,self.BIDevH), min(self.LDevH,self.DPDevL))
        self.FeedConst=-self.FeedAtLowConst+self.FeedAtHighConst
        
        #L restricted for quality only not for material balance
        self.LAtLowConst=max(self.LDevL,self.DPDevL)
        if GapMult==1:
            self.LAtHighConst=max(self.LDevH,self.BIDevH)
        else:
            self.LAtHighConst=max(self.LDevH,BIDevHAdj)
        self.LConst=-self.LAtLowConst+self.LAtHighConst
        
        #V restricted for quality only not for material balance
        self.VAtLowConst=max(self.VDevL,self.BIDevH) 
        if GapMult==1:
            self.VAtHighConst=max(self.VDevH,self.DPDevL)
        else:
            self.VAtHighConst=max(self.VDevH,DPDevLAdj)
        self.VConst=-self.VAtLowConst+self.VAtHighConst
      
        if self.FeedConst>0:
            nFeedConst=1*abs(self.ConstFactor)
        if self.FeedConst<0:
            nFeedConst=-1*abs(self.ConstFactor)
        if self.FeedConst==0:
            nFeedConst=0
        if self.LConst>0:
            nLConst=1*abs(self.ConstFactor)
        if self.LConst<0:
            nLConst=-1*abs(self.ConstFactor)
        if self.LConst==0:
            nLConst=0
        if self.VConst>0:
            nVConst=1*abs(self.ConstFactor)
        if self.VConst<0:
            nVConst=-1*abs(self.ConstFactor)
        if self.VConst==0:
            nVConst=0

        State=np.array([nFeedConst,nLConst,nVConst])
        return State
       
    def RewardCalc(self):
        self.PrevValue=self.NewValue
        self.altPrevValue=self.altNewValue
        if self.distpurityPred<=self.targ_distpurity:
            self.distPurityContrib=pow(self.distpurityPred/(self.targ_distpurity+self.DPGap),5)
        else:
            self.distPurityContrib=1
        if self.btmimpurityPred>=self.targ_btmimpurity:
            self.btmImpurityContrib=pow((100-self.btmimpurityPred)/(100-self.targ_btmimpurity+self.BIGap),5) 
        else:
            self.btmImpurityContrib=1
        self.ConstContrib=(abs(self.FeedConst)+abs(self.LConst)+abs(self.VConst))*self.ConstFactor
        self.profit=((self.Dval*self.DPrice*self.distPurityContrib+self.Bval*self.BPrice*self.btmImpurityContrib)- \
                     (self.Feedval*self.FeedPrice+self.Lval*self.LPrice+self.Vval*self.VPrice))*self.ProfitFactor
        self.NewValue=self.profit+self.ConstContrib
        self.reward=self.NewValue-self.PrevValue
        # calculate alternate reward for alternate limits as mentioned in section 4.4 in the paper
        self.altprofit=((self.Dval*self.DPrice+self.Bval*self.BPrice)- \
                     (self.Feedval*self.FeedPrice+self.Lval*self.LPrice+self.Vval*self.VPrice))*self.ProfitFactor
        self.altNewValue=self.altprofit
        self.altreward=self.altNewValue-self.altPrevValue
        
        return self.reward, self.altreward

    def RestrictActionCalc(self):
        # code to restrict actions when in safe zone
        self.StateCalc(2)   # bigger gap to include safe zone
        self.RestrictAction=np.array([0,0,0])
        
        if self.FeedAtHighConst>0:
            self.RestrictAction[0]=1    #Feed inc not allowed
        if self.FeedAtLowConst>0:
            self.RestrictAction[0]=-1    #Feed dec not allowed
        if self.FeedAtHighConst>0 and self.FeedAtLowConst>0:
            self.RestrictAction[0]=2    #Feed change not allowed
            
        if self.LAtHighConst>0:
            self.RestrictAction[1]=1    #L inc not allowed
        if self.LAtLowConst>0:
            self.RestrictAction[1]=-1    #L dec not allowed
        if self.LAtHighConst>0 and self.LAtLowConst>0:
            self.RestrictAction[1]=2    #L change not allowed
            
        if self.VAtHighConst>0:
            self.RestrictAction[2]=1    #V inc not allowed
        if self.VAtLowConst>0:
            self.RestrictAction[2]=-1    #V dec not allowed
        if self.VAtHighConst>0 and self.VAtLowConst>0:
            self.RestrictAction[2]=2    #V change not allowed

    def SuppressActionCalc(self):

        if (self.btmimpurity<=self.targ_btmimpurity+self.BIGap) and (self.btmimpurity>=self.targ_btmimpurity-self.BIGap*3):
            self.BInl=(1==1)
        else:
            self.BInl=(1!=1)
        if (self.distpurity>=self.targ_distpurity-self.DPGap) and (self.distpurity<=self.targ_distpurity+self.DPGap*3):
            self.DPnl=(1==1)
        else:
            self.DPnl=(1!=1)
        if (self.Dval<=self.D_HL+self.DGap and self.Dval>=self.D_HL-self.DGap*3) or \
        (self.Dval>=self.D_LL-self.DGap) and (self.Dval<=self.D_LL+self.DGap*3):
            self.Dnl=(1==1)
        else:
            self.Dnl=(1!=1)
        if (self.Bval<=self.B_HL+self.BGap and self.Bval>=self.B_HL-self.BGap*3) or \
        (self.Bval>=self.B_LL-self.BGap) and (self.Bval<=self.B_LL+self.BGap*3):
            self.Bnl=(1==1)
        else:
            self.Bnl=(1!=1)
        if (self.Lval<=self.L_HL+self.LGap and self.Lval>=self.L_HL-self.LGap*3) or \
        (self.Lval>=self.L_LL-self.LGap) and (self.Lval<=self.L_LL+self.LGap*3):
            self.Lnl=(1==1)
        else:
            self.Lnl=(1!=1)
        if (self.Vval<=self.V_HL+self.VGap and self.Vval>=self.V_HL-self.VGap*3) or \
        (self.Vval>=self.V_LL-self.VGap) and (self.Vval<=self.V_LL+self.VGap*3):
            self.Vnl=(1==1)
        else:
            self.Vnl=(1!=1)
        if (self.Feedval<=self.Feed_HL+self.FeedGap and self.Feedval>=self.Feed_HL-self.FeedGap*3) or \
        (self.Feedval>=self.Feed_LL-self.FeedGap) and (self.Feedval<=self.Feed_LL+self.FeedGap*3):
            self.Feednl=(1==1)
        else:
            self.Feednl=(1!=1)
        self.suppressaction=self.BInl or self.DPnl or self.Dnl or self.Bnl or self.Feednl or self.Lnl or self.Vnl
            
        