import os
import glob
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

plt.close("all");
plt.style.use('classic');
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 24};
plt.rc('font', **font);

mpl.rcParams['xtick.major.size'] = 5;
mpl.rcParams['xtick.major.width'] = 2;
mpl.rcParams['xtick.minor.size'] = 2.5;
mpl.rcParams['xtick.minor.width'] = 2;

mpl.rcParams['ytick.major.size'] = 5;
mpl.rcParams['ytick.major.width'] = 2;
mpl.rcParams['ytick.minor.size'] = 2.5;
mpl.rcParams['ytick.minor.width'] = 2;

os.system('grep -r "Train\["  nekRS-ML/nekrs/nekrs.out > z_nekrs.txt')
os.system('grep -r "Training set: Average loss"  nekRS-ML/train/train.out > z_error.txt')

label_cfd = f"zcfd"
label = f"zerror"
input_cfd = np.loadtxt("z_nekrs.txt", usecols=(3,4,))
input_tmp = np.loadtxt("z_error.txt", usecols=(4,))

nepoch = len(glob.glob1('nekRS-ML/train/',"*.csv"))

##--------------------------------------##
##--------------------------------------##
yy = np.sqrt(input_cfd[:,0])
xx = np.linspace(0,len(yy),len(yy))+1;

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
plt.plot(xx,yy,'k')
#plt.xlabel('Iteration')
plt.ylabel(r'$\overline{u}_{\tau}/U_o$')
plt.ylim((0.9*np.amin(yy),1.1*np.amax(yy)))
plt.savefig(f'{label_cfd}.pdf',format="pdf")

yy = input_cfd[:,1]
xx = np.linspace(0,len(yy),len(yy))+1;
ax = fig.add_subplot(2, 1, 2)
plt.plot(xx,yy,'k')
plt.xlabel('Iteration')
plt.ylabel(r'$\overline{U}_{t}^{s}/U_o$')
plt.ylim((0.9*np.amin(yy),1.1*np.amax(yy)))
plt.tight_layout()
plt.savefig(f'{label_cfd}.pdf',format="pdf")

##--------------------------------------##
##--------------------------------------##
yy = input_tmp[:]
yy = np.divide(yy,yy[0])
xx = np.linspace(0,len(yy),len(yy))+1;


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(xx,yy,'k-o')
plt.xlabel('Epoch')
plt.ylabel('Error')
ax.set_yscale('log')
#plt.grid(True)
#plt.xlim((0,2*np.pi))
#plt.ylim((-1,1))
plt.tight_layout()
plt.savefig(f'{label}.pdf',format="pdf")

##--------------------------------------##
##--------------------------------------##
if(nepoch):
  for i,epoch in enumerate(range(1,nepoch-1,10)):
    residuals = np.loadtxt(f"nekRS-ML/train/residuals_epoch_{epoch}.csv", delimiter=",")
    weights = (1/len(residuals))*np.ones(np.shape(residuals))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(residuals, bins=30, weights=weights, density=False, stacked=False)
    plt.title(f"Residual Distribution - Epoch {epoch}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.ylim((0,0.2))
    plt.savefig(f"zresidual_hist_epoch_{i}.png")
    plt.close()
