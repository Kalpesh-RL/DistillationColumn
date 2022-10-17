import matplotlib.pyplot as plt
import numpy as np

# function to trend absolute value of individual actions along with its limits and zones
def actiontrend(plotrow, plotcol,plotindex, varname,stepindex, time_sequence, var_sequence, var_LL, var_HL, var_Gap,disphours,simtime):
    ax=plt.subplot(plotrow, plotcol,plotindex)
    ax.grid()    
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var_LL*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'-',color="black")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var_HL*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'-',color="black")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_LL+var_Gap)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="red")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_HL-var_Gap)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="red")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_LL+var_Gap*2)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="green")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_HL-var_Gap*2)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="green")
    #plt.xlabel("time - hours")
    plt.ylabel(varname)
    plt.ylim(var_LL-var_Gap,var_HL+var_Gap)

# function to trend a limit variable with a low limit
def LVLtrend(plotrow, plotcol,plotindex, varname,stepindex, time_sequence, var_sequence, varpred_sequence,target_sequence, LL,HL,var_Gap,disphours,simtime):
    ax=plt.subplot(plotrow, plotcol,plotindex)
    ax.grid()    
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,varpred_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,target_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]+0,'-',color="black")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,target_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]+var_Gap,'--',color="red")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,target_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]+var_Gap*2,'--',color="green")
    """
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(target)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="black")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(target+var_Gap)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="red")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(target+var_Gap*2)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="green")
    """
    #plt.xlabel("time - hours")
    plt.ylabel(varname)
    plt.ylim(LL,HL)

# function to trend a limit variable with a high limit
def LVHtrend(plotrow, plotcol,plotindex, varname,stepindex, time_sequence, var_sequence, varpred_sequence,target, LL,HL,var_Gap,disphours,simtime):
    ax=plt.subplot(plotrow, plotcol,plotindex)
    ax.grid()    
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,varpred_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(target)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'-',color="black")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(target-var_Gap)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="red")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(target-var_Gap*2)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="green")
    #plt.xlabel("time - hours")
    plt.ylabel(varname)
    plt.ylim(LL,HL)

# function to trend a limit variable with a high & low limit
def LVtrend(plotrow, plotcol,plotindex, varname,stepindex, time_sequence, var1_sequence, var2_sequence, var_LL,var_HL,var_Gap,disphours,simtime):
    ax=plt.subplot(plotrow, plotcol,plotindex)
    ax.grid()    
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var1_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var2_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var_LL*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'-',color="black")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var_HL*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'-',color="black")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_LL+var_Gap)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="red")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_HL-var_Gap)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="red")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_LL+var_Gap*2)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="green")
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,(var_HL-var_Gap*2)*np.ones((min(stepindex+1,int(disphours*60/simtime)))),'--',color="green")
    #plt.xlabel("time - hours")
    plt.ylabel(varname)
    plt.ylim(var_LL-var_Gap,var_HL+var_Gap)

# function to trend 3 variables
def trend3(plotrow, plotcol,plotindex, varname,stepindex, time_sequence, var1_sequence, var2_sequence,var3_sequence, disphours,simtime,LL=0, HL=0):
    ax=plt.subplot(plotrow, plotcol,plotindex)
    ax.grid()    
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var1_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var2_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    plt.plot(time_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1]/60,var3_sequence[max(stepindex+1-int(disphours*60/simtime),0):stepindex+1])
    #plt.xlabel("time - hours")
    plt.ylabel(varname)
    if LL!=0 or HL!=0:
        plt.ylim(LL,HL)

