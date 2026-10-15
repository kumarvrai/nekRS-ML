import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

#------------------------------------#

case = sys.argv[1]
tStart = convert_to_float(sys.argv[2])
#visco    = convert_to_float(sys.argv[2]);
#Lz       = convert_to_float(sys.argv[3]);
#strtTime = convert_to_float(sys.argv[4]);
#aoa      = convert_to_float(sys.argv[5]);

os.system('grep -a "t-min-max" nekrs.out | sed "/<</d" > t_hist_minmax.txt')
print('--||NRS :: LOADING UVWP DATA')
if('RANS' in case):
  col_to_read=np.arange(1,17,1, dtype=int)
  nplots = [1, 3, 5, 7, 11, 13];
  ylab = [r'$u/U_o$',r'$v/U_o$',r'$w/U_o$',r'$p/\rho U_o^2$', r'$k/U_o^2$', r'$\omega c/U_o$']
else:
  col_to_read=np.arange(1,12,1, dtype=int)
  nplots = [1, 3, 5, 7, 9];
  ylab = [r'$max(U)$',r'$max(V)$',r'$max(W)$',r'$max(P)$',r'$max(\nu)$']

minmax_data = np.loadtxt('t_hist_minmax.txt', skiprows=100,usecols = col_to_read)

indTime = 0;
time = minmax_data[:,indTime]
print('--||NRS :: TIME RANGE',np.amin(time),np.amax(time))
startInd = np.argmin(abs(time-tStart),axis=None);
minmax_data = minmax_data[startInd::,:]
time = minmax_data[:,indTime]

print('--||NRS :: PLOTTING UVWP HISTORY')
fig, axs = plt.subplots(int(np.ceil(len(nplots)/2)), 2, sharex=True, dpi=300)
ax=axs.flatten()

for i,ind in enumerate(nplots):
  ax[i].plot(time,minmax_data[:,ind],linewidth=1.5)
  ax[i].plot(time,minmax_data[:,ind+1],linewidth=1.5)
  ax[i].set_xlabel(r'$t$')
  ax[i].set_ylabel(ylab[i])

plt.tight_layout()
plt.savefig('nrs_analysis.png')
#plt.savefig('nrs_analysis.pdf')
#plt.show()

if('CLCD' in case):
  os.system('grep "t-cd-cl" nekrs.out | sed "/<</d" > t_hist_minmax.txt')
  print('--||NRS :: LOADING CLCD DATA')
  col_to_read=np.arange(1,8,1, dtype=int)
  nplots = [3, 6];
  ylab = [r'$C_d$',r'$C_l$']
  
  minmax_data = np.loadtxt('t_hist_minmax.txt', skiprows=100,usecols = col_to_read)
  
  indTime = 0;
  time = minmax_data[:,indTime]
  
  print('--||NRS :: PLOTTING CLCD HISTORY')
  fig, axs = plt.subplots(len(nplots), 1, sharex=True, dpi=300)
  ax=axs.flatten()
  
  for i,ind in enumerate(nplots):
    ax[i].plot(time,minmax_data[:,ind],linewidth=1.5)
    ax[i].set_xlabel(r'$t$')
    ax[i].set_ylabel(ylab[i])
  
  plt.tight_layout()
  plt.savefig('nrs_clcd.png')
if('MONITOR' in case):
  os.system('grep -a "monitor" nekrs.out | sed "/<</d" > t_hist_minmax.txt')
  nplots = [2, 3, 4, 5, 6, 7];
  #ubpow  = [1, 2, 1, 1];
  ubpow  = [0, 0, 0, 0, 0, 0];
  #ylab = [r'$U_{b}$',r'$\Sigma U_i^2/U_b^2$',r'$u_{\tau}/U_b$',r'$Re_{\tau}$']
  ylab = [r'$U_{b}$',r'$\Sigma U_i^2$',r'$F_v$',r'$F_p$',r'$u_{\tau}$',r'$Re_{\tau}$']
  col_to_read=np.arange(1,9,1, dtype=int)
  print('--||NRS :: LOADING MONITOR DATA')
  minmax_data = np.loadtxt('t_hist_minmax.txt', skiprows=100,usecols = col_to_read)
  
  indTime = 1;
  time = minmax_data[:,indTime]
  print('--||NRS :: TIME RANGE',np.amin(time),np.amax(time))
  startInd = np.argmin(abs(time-tStart),axis=None);
  minmax_data = minmax_data[startInd::,:]
  time = minmax_data[:,indTime]
  ubulk = np.mean(minmax_data[:,nplots[0]],axis=None)
  utau  = np.mean(minmax_data[:,nplots[4]],axis=None)
  utub  = np.mean(np.divide(minmax_data[:,nplots[4]],minmax_data[:,nplots[0]]),axis=None)
  print('--||NRS :: AVERAGE UBULK = %5.2E'%ubulk)
  print('--||NRS :: AVERAGE UTAU  = %5.2E'%utau)
  print('--||NRS :: AVERAGE UT/UB  = %5.2E'%utub)

  print('--||NRS :: PLOTTING MONITOR DATA')
  fig, axs = plt.subplots(int(len(nplots)/2), 2, sharex=True, dpi=300)
  ax=axs.flatten()
  
  for i,ind in enumerate(nplots):
    plotData = minmax_data[:,ind]/(ubulk**ubpow[i]);
    ax[i].plot(time,plotData,linewidth=1.5)
    ax[i].set_xlabel(r'$t$')
    ax[i].set_ylabel(ylab[i])
  
  plt.tight_layout()
  plt.savefig('nrs_monitor.png')
