import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

os.system('grep -r "wall-pts"  nekRS-ML/nekrs/nekrs.out > wall.txt')
os.system('grep -r "offw-pts"  nekRS-ML/nekrs/nekrs.out > owall.txt')

plttype = '2D'
input_tmp = np.loadtxt("wall.txt", usecols=(1,2,3,), skiprows=2)

pltSkip = 2
xx = input_tmp[::pltSkip,0]
yy = input_tmp[::pltSkip,1]
zz = input_tmp[::pltSkip,2]

input_tmp = np.loadtxt("owall.txt", usecols=(1,2,3,), skiprows=2)

xx2 = input_tmp[::pltSkip,0]
yy2 = input_tmp[::pltSkip,1]
zz2 = input_tmp[::pltSkip,2]

label = f"interp_nodes"
colors = ['b', 'r','g','k']

if(plttype == '2D'):
  indSort = np.where(zz==0.0)
  xx = xx[indSort];
  yy = yy[indSort];
  xx2 = xx2[indSort];
  yy2 = yy2[indSort];
  plt.scatter(xx,yy,s=10,c=colors[0])
  plt.scatter(xx2,yy2,s=10,c=colors[1])
  for i in range(0, len(xx), 1):
    #plt.plot(xx[i:i+2], yy[i:i+2], 'ro-')
    plt.arrow(xx[i], yy[i], xx2[i]-xx[i], yy2[i]-yy[i])
  plt.xlabel(r'$x/H$', fontsize=32)
  plt.ylabel(r'$y/H$', fontsize=32)
  #plt.grid(True)
else:  
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(xx,yy,zz,s=1,c=colors[0])
  ax.scatter(xx2,yy2,zz2,s=1,c=colors[1])
  ax.set_xlabel('X-axis')
  ax.set_ylabel('Y-axis')
  ax.set_zlabel('Z-axis')
  #ax.set_xlim((0,9.0))
  #ax.set_ylim((0,3.035))
  #ax.set_zlim((0,4.5))
  ax.view_init(elev=90, azim=-90, roll=0)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=16)  
plt.tight_layout()
plt.savefig(f'{label}.png',dpi=300)
#plt.savefig(f'{label}.pdf',format="pdf")
print(f"Saved file with Label: {label}")                                 

