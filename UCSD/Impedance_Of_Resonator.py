import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Initialize variables
Ls = 20e-12 #SQUID self inductance
Lj = 60e-12 #SQUID josephson inductance
Cj = 100e-15 #SQUID junction capacitance
Rsg = 100 #SQUID sub gap loss
Mc = 1e-12 #Resonator to SQUID mutual inductance
Lc = 80e-12 #SQUID couple inductance
Cc = 100e-12 #Resonator to t-line coupling capacitance
Z1 = 50
omega_r = 2*np.pi*5e9
pi = np.pi

Omegas = 2*pi*np.linspace(4.8e9,4.9e9,1000)
phis = np.linspace(-pi/2,5*pi/2,100)

Zeff = np.empty((len(phis),len(Omegas)),dtype=complex)
Z_loaded_tline = np.empty((len(phis),len(Omegas)),dtype=complex)
Ztot = np.empty((len(phis),len(Omegas)),dtype=complex)

#Now define some derived quantities
Ys = 1/Rsg + 1j*Omegas*Cj #Shunt admittance
ys = 1j*Omegas*Lj*Ys
#Ztline = 1j*50*np.tan(Omegas*(pi/(4*pi*omega_r)))
ZCc = 1/(1j*Omegas*Cc)

max_val = []
for i,phi in enumerate(phis):
  Zeff[i,:] = 1j*Omegas*(Lc-(Mc**2)/(Ls+(Lj/(np.cos(phi)+ys))))
  Z_loaded_tline[i,:] = Z1*(Zeff[i,:] + 1j*Z1*np.tan(Omegas*pi/(2*omega_r)))/(Z1 + 1j*Zeff[1,:]*np.tan(Omegas*pi/(2*omega_r)))
  Ztot[i,:] = ZCc + Z_loaded_tline[i,:]

  max_val.append(Omegas[np.argmax(np.abs(Ztot[i,:]))])
  plt.figure(7)
  plt.plot(phi,np.max(np.abs(Ztot[i,:])),'o')

phis,Omegas = np.meshgrid(phis,Omegas)
fig = plt.figure()
ax = plt.axes(projection='3d',zscale = 'log')
ax.contour3D(phis,Omegas/(2*pi)/1e9,np.transpose(np.abs(Ztot)),100,cmap='viridis')

plt.show()
#plt.figure(1)
#plt.plot(Omegas,np.abs(Ztot[i,:]))

#for i,Omega in enumerate(Omegas):
#  plt.figure(1)
#  plt.plot(phis,(np.abs(Ztot[:,i])-np.min(np.abs(Ztot[:,i])))/np.max(np.abs(Ztot[:,i]))*100)
#plt.ylabel('% Change in Impedance')
#plt.show()
