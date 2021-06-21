import numpy as np
from numpy import pi
import scipy.optimize as op
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
plt.ion()

# Note: lambda is a reserved word, so I use lamb instead
Phi0 = 2.068e-15 # magnetic flux quantum

def phieofphi(phi,lamb):
  phie = phi + lamb*np.sin(phi)
  return phie

def phisum(phi,*args):
    phie,lamb = args
    return (phieofphi(phi,lamb) - phie)

def phiofphie(phie, lamb):
    phiguess = phie
    argvals = (phie,lamb)
    phi = fsolve(phisum, phiguess, args=argvals)
    return phi

def f0ofphi(phi, f2, P, lamb):
    # Odd formulation hopefully makes variation in lamb not affect f2 or P
    f0 = f2 + (P/2) * ((1-lamb**2)/lamb) * ((lamb*np.cos(phi))/(1 + lamb*np.cos(phi)) + (lamb**2)/(1-lamb**2))
    return f0

def f0ofphie(phie, f2, P, lamb):
    phi = phiofphie(phie,lamb)
    f0 = f0ofphi(phi,f2,P,lamb)
    return f0

p = np.linspace(0,2*np.pi,1000)

#Plot what the curves look like for fixed df_pp
#and varying lambda
ncol = np.linspace(0, 1, 10)
colors = [cm.jet(x) for x in ncol]
for i,l in enumerate(np.linspace(0.01,0.9,10)):
    plt.plot(p/(2*pi),(f0ofphie(p,6e9,100e3,l)-6e9)/1e3,color = colors[i],
    label = f'$\lambda$ = {np.round(l,2)}')
plt.legend()
plt.ylim(-60,60)
plt.yticks(np.arange(-60,70,10))
plt.grid()
plt.xlabel('$N_{\Phi_0}$ ',fontsize = 24)
plt.ylabel('df [kHz]',fontsize = 24)
fig = plt.gcf()
fig.set_size_inches(11,8)
plt.savefig('/home/msilvafe/Documents/vary_lambda.png')
plt.close()

#Plot what the derivative of the curves look like for fixed df_pp
#and varying lambda and the cumsum overplotted to see what
#is contributing most to the integral
plt.figure()
ax1 = plt.gca()
ax2 = ax1.twinx()
for i,l in enumerate(np.linspace(0.01,0.9,10)):
    ax1.semilogy(p[0:-1]/(2*pi),
        (np.diff(f0ofphie(p,6e9,100e3,l))/np.diff(p))**2,
        color = colors[i],label = f'$\lambda$ = {np.round(l,2)}')
    ax2.semilogy(p[0:-1]/(2*pi),
    np.cumsum((np.diff(f0ofphie(p,6e9,100e3,l))/np.diff(p))**2),
    '--',color = colors[i])
ax1.legend()
ax1.set_xlabel('$N_{\Phi_0}$ ',fontsize = 24)
ax1.set_ylabel('$(df/d\phi)^2$ [(Hz/Radian)$^2$]',fontsize = 24)
fig = plt.gcf()
fig.set_size_inches(11,8)
ax2.set_ylabel('Cummulative Sum',fontsize = 24)
plt.savefig('/home/msilvafe/Documents/deriv_sqr_vary_lambda_cumsum.png')
plt.close()

df_dI_arr = np.zeros((10,12))
for i,pp in enumerate(np.linspace(20,160,num = 10)):
    for j,l in enumerate(np.linspace(0.1,0.9,num=12)):
        df_dI_arr[i,j] = np.sqrt(np.sum((np.diff(f0ofphie(p,6e9,pp,l))/np.diff(p))**2)*(p[1]-p[0])/(2*np.pi))*(2*pi/2.067833848e-15)*227e-12*1e-12
X,Y = np.meshgrid(np.linspace(20,160,num=10),
    np.linspace(0.1,0.9,num=12))
levels = np.linspace(np.nanmin(df_dI_arr*1e6),
    np.nanmax(df_dI_arr*1e6),num = 100)
cs = plt.contourf(X,Y,df_dI_arr*1e6,levels=levels,cmap = 'RdGy')
cbar = plt.colorbar(cs)
plt.ylabel('$\lambda$',fontsize = 24)
plt.xlabel('$df_{pp}$ [kHz]',fontsize = 24)
cbar.set_label('$df/dI$ [pA/$\Phi_0$]',fontsize = 24)
fig = plt.gcf()
fig.set_size_inches(11,8)
plt.savefig('/home/msilvafe/Documents/dfdI_vs_dfpp_vs_lambda.png')
plt.close()
