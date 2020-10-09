import numpy as np
import matplotlib.pyplot as plt

def plot_well_feature(x_trainwell, test_loc, test_inc, d):
    
    interval1 = [test_loc[0]*d+ 200,(test_loc[0]+test_inc)*d+ 200]

    interval2 = [test_loc[1]*d+ 200,(test_loc[1]+test_inc)*d+ 200] 
    
    
    depth = d * np.linspace(0.0, len(x_trainwell), num = len(x_trainwell)) + 200
    
    fig = plt.figure(figsize=(12, 8))
    fig.set_facecolor('white')
    ax1 = fig.add_axes([0.06, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax2 = fig.add_axes([0.19, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax3 = fig.add_axes([0.32, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax4 = fig.add_axes([0.45, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax5 = fig.add_axes([0.58, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax6 = fig.add_axes([0.71, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    ax7 = fig.add_axes([0.84, 0.1, 0.1, 0.8],
                       ylim=(depth[0] ,depth[-1]))
    
    ax1.plot(x_trainwell[:,0],depth,'-k', lw = 2)
    ax1.fill_between([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax1.fill_between([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax1.set_xlim([np.min(x_trainwell[:,0])-0.1*np.min(x_trainwell[:,0]),np.max(x_trainwell[:,0])+0.1*np.min(x_trainwell[:,0])])
    ax1.invert_yaxis()
    ax1.set_xlabel('CAL',fontsize= 14)
    ax1.set_ylabel('Depth (m)',fontsize = 14)
    ax1.grid(linestyle='-.',linewidth=1.5)
    ax1.tick_params(labelsize = 12)  
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    
    ax2.plot(x_trainwell[:,1],depth,'-k', lw = 2)
    ax2.fill_between([np.min(x_trainwell[:,1])-0.1*np.min(x_trainwell[:,1]),np.max(x_trainwell[:,1])+0.1*np.min(x_trainwell[:,1])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax2.fill_between([np.min(x_trainwell[:,1])-0.1*np.min(x_trainwell[:,1]),np.max(x_trainwell[:,1])+0.1*np.min(x_trainwell[:,1])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax2.set_xlim([np.min(x_trainwell[:,1])-0.1*np.min(x_trainwell[:,1]),np.max(x_trainwell[:,1])+0.1*np.min(x_trainwell[:,1])])
    ax2.invert_yaxis()
    ax2.set_xlabel('PHI',fontsize= 14)
    ax2.grid(linestyle='-.',linewidth=1.5)
    ax2.tick_params(labelsize = 12)  
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)
    
    ax3.plot(x_trainwell[:,2],depth,'-k', lw = 2)
    ax3.fill_between([np.min(x_trainwell[:,2])-0.1*np.min(x_trainwell[:,2]),np.max(x_trainwell[:,2])+0.1*np.min(x_trainwell[:,2])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax3.fill_between([np.min(x_trainwell[:,2])-0.1*np.min(x_trainwell[:,2]),np.max(x_trainwell[:,2])+0.1*np.min(x_trainwell[:,2])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax3.set_xlim([np.min(x_trainwell[:,2])-0.1*np.min(x_trainwell[:,2]),np.max(x_trainwell[:,2])+0.1*np.min(x_trainwell[:,2])])
    ax3.invert_yaxis()
    ax3.set_xlabel('GR',fontsize= 14)
    ax3.grid(linestyle='-.',linewidth=1.5)
    ax3.tick_params(labelsize = 12)  
    ax3.spines['bottom'].set_linewidth(1.5)
    ax3.spines['left'].set_linewidth(1.5)
    ax3.spines['right'].set_linewidth(1.5)
    ax3.spines['top'].set_linewidth(1.5)
    
    ax4.plot(x_trainwell[:,3],depth,'-k', lw = 2)
    ax4.fill_between([np.min(x_trainwell[:,3])-0.1*np.min(x_trainwell[:,3]),np.max(x_trainwell[:,3])+0.1*np.min(x_trainwell[:,3])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax4.fill_between([np.min(x_trainwell[:,3])-0.1*np.min(x_trainwell[:,3]),np.max(x_trainwell[:,3])+0.1*np.min(x_trainwell[:,3])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax4.set_xlim([np.min(x_trainwell[:,3])-0.1*np.min(x_trainwell[:,3]),np.max(x_trainwell[:,3])+0.1*np.min(x_trainwell[:,3])])
    ax4.invert_yaxis()
    ax4.set_xlabel('DR',fontsize= 14)
    ax4.grid(linestyle='-.',linewidth=1.5)
    ax4.tick_params(labelsize = 12)  
    ax4.spines['bottom'].set_linewidth(1.5)
    ax4.spines['left'].set_linewidth(1.5)
    ax4.spines['right'].set_linewidth(1.5)
    ax4.spines['top'].set_linewidth(1.5)
    
    ax5.plot(x_trainwell[:,4],depth,'-k', lw = 2)
    ax5.fill_between([np.min(x_trainwell[:,4])-0.1*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.1*np.min(x_trainwell[:,4])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax5.fill_between([np.min(x_trainwell[:,4])-0.1*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.1*np.min(x_trainwell[:,4])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax5.set_xlim([np.min(x_trainwell[:,4])-0.1*np.min(x_trainwell[:,4]),np.max(x_trainwell[:,4])+0.1*np.min(x_trainwell[:,4])])
    ax5.invert_yaxis()
    ax5.set_xlabel('MR',fontsize= 14)
    ax5.grid(linestyle='-.',linewidth=1.5)
    ax5.tick_params(labelsize = 12)  
    ax5.spines['bottom'].set_linewidth(1.5)
    ax5.spines['left'].set_linewidth(1.5)
    ax5.spines['right'].set_linewidth(1.5)
    ax5.spines['top'].set_linewidth(1.5)
    
    ax6.plot(x_trainwell[:,5],depth,'-k', lw = 2)
    ax6.fill_between([np.min(x_trainwell[:,5])-0.1*np.min(x_trainwell[:,5]),np.max(x_trainwell[:,5])+0.1*np.min(x_trainwell[:,5])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax6.fill_between([np.min(x_trainwell[:,5])-0.1*np.min(x_trainwell[:,5]),np.max(x_trainwell[:,5])+0.1*np.min(x_trainwell[:,5])],[interval2[0]],[interval2[1]],color="red",alpha=0.7)
    ax6.set_xlim([np.min(x_trainwell[:,5])-0.1*np.min(x_trainwell[:,5]),np.max(x_trainwell[:,5])+0.1*np.min(x_trainwell[:,5])])
    ax6.invert_yaxis()
    ax6.set_xlabel('PE',fontsize= 14)
    ax6.grid(linestyle='-.',linewidth=1.5)
    ax6.tick_params(labelsize = 12)  
    ax6.spines['bottom'].set_linewidth(1.5)
    ax6.spines['left'].set_linewidth(1.5)
    ax6.spines['right'].set_linewidth(1.5)
    ax6.spines['top'].set_linewidth(1.5)
    
    ax7.plot(x_trainwell[:,6],depth,'-k', lw = 2)
    ax7.fill_between([np.min(x_trainwell[:,6])-0.1*np.min(x_trainwell[:,6]),np.max(x_trainwell[:,6])+0.1*np.min(x_trainwell[:,6])],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax7.fill_between([np.min(x_trainwell[:,6])-0.1*np.min(x_trainwell[:,6]),np.max(x_trainwell[:,6])+0.1*np.min(x_trainwell[:,6])],[interval2[0]],[interval2[1]],color="red",alpha=0.7,label="Testing")
    ax7.set_xlim([np.min(x_trainwell[:,6])-0.1*np.min(x_trainwell[:,6]),np.max(x_trainwell[:,6])+0.1*np.min(x_trainwell[:,6])])
    ax7.invert_yaxis()
    ax7.set_xlabel('RHO',fontsize= 14)
    ax7.grid(linestyle='-.',linewidth=1.5)
    ax7.tick_params(labelsize = 12)  
    ax7.spines['bottom'].set_linewidth(1.5)
    ax7.spines['left'].set_linewidth(1.5)
    ax7.spines['right'].set_linewidth(1.5)
    ax7.spines['top'].set_linewidth(1.5)
    # ax7.legend(bbox_to_anchor=(0.9, 1))
    
def plot_well_target(y_trainwell, test_loc, test_inc, d):
    
    interval1 = [test_loc[0]*d+ 200,(test_loc[0]+test_inc)*d+ 200]

    interval2 = [test_loc[1]*d+ 200,(test_loc[1]+test_inc)*d+ 200] 
    
    
    depth = d * np.linspace(0.0, len(y_trainwell), num = len(y_trainwell)) + 200
    

    fig = plt.figure(figsize=(4, 8))
    fig.set_facecolor('white')
    ax1 = fig.add_axes([0.2, 0.1, 0.31, 0.8],
                    ylim=(depth[0] ,depth[-1]))
    
    ax1.plot(y_trainwell,depth,'-k', lw = 2)
    ax1.fill_between([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax1.fill_between([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)],[interval2[0]],[interval2[1]],color="red",alpha=0.7,label="Testing")
    ax1.set_xlim([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)])
    ax1.invert_yaxis()
    ax1.set_xlabel('DTS',fontsize= 14)
    ax1.set_ylabel('Depth (m)',fontsize = 14)
    ax1.grid(linestyle='-.',linewidth=1.5)
    ax1.tick_params(labelsize = 12)  
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    

def plot_well_predict(y_trainwell, Y_predict, test_loc, test_inc, d):
    
    interval1 = [test_loc[0]*d+ 200,(test_loc[0]+test_inc)*d+ 200]

    interval2 = [test_loc[1]*d+ 200,(test_loc[1]+test_inc)*d+ 200] 
    
    
    depth = d * np.linspace(0.0, len(y_trainwell), num = len(y_trainwell)) + 200
    

    fig = plt.figure(figsize=(4, 8))
    fig.set_facecolor('white')
    ax1 = fig.add_axes([0.2, 0.1, 0.31, 0.8],
                    ylim=(depth[0] ,depth[-1]))
    
    ax1.plot(y_trainwell,depth,'-r', lw = 2)
    ax1.plot(Y_predict,depth,'-b', lw = 2)
    ax1.fill_between([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)],[interval1[0]],[interval1[1]],color="red",alpha=0.7)
    ax1.fill_between([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)],[interval2[0]],[interval2[1]],color="red",alpha=0.7,label="Testing")
    ax1.set_xlim([np.min(y_trainwell)-0.1*np.min(y_trainwell),np.max(y_trainwell)+0.1*np.min(y_trainwell)])
    ax1.invert_yaxis()
    ax1.set_xlabel('DTS',fontsize= 14)
    ax1.set_ylabel('Depth (m)',fontsize = 14)
    ax1.grid(linestyle='-.',linewidth=1.5)
    ax1.tick_params(labelsize = 12)  
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)
    
    
def plot_pred_interval(model, Y_test, X_test, test_loc, test_inc, d):    
    
    from pred_ints import pred_ints
    
    for i in range(len(test_loc)):
    
        depth = d * np.linspace(0.0, test_inc, num = int(test_inc)) + test_loc[i] * d + 200
        
        X_input = X_test[(i)*test_inc:(i+1)*test_inc,:]
        
        A = pred_ints(model, X_input, 90)        
        A = np.array(A)
        
        A1 = pred_ints(model, X_input, 60)
        A1 = np.array(A1)
        
        A2 = pred_ints(model, X_input, 30)
        A2 = np.array(A2)
        
        
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot()
        fig.set_facecolor('white')
        plt.plot(depth,model.predict(X_input),'b',label="Prediction")
        plt.plot(depth,Y_test[(i)*test_inc:(i+1)*test_inc],'r',label="Reference")
        plt.fill_between(depth,A[0,:],A[1,:], color='b', alpha=.1,label="90%")       
        plt.fill_between(depth,A1[0,:],A1[1,:], color='r', alpha=.1,label="60%")       
        plt.fill_between(depth,A2[0,:],A2[1,:], color='k', alpha=.1,label="30%")
        plt.legend(loc='best')
        ax.set_xlabel('Depth (m)',fontsize= 14)
        ax.set_ylabel('DTS (μs/ft)',fontsize = 14)           
        ax.grid(linestyle='-.',linewidth=1.5)
        ax.tick_params(labelsize = 12)  
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        
            

    
