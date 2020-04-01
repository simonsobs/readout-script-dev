import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
pi = np.pi

def f(y, t, params):
    I, Idot = y      # unpack current values of y
    L0, R, C, gamma, omega_m, t0, V0 = params  # unpack parameters
    derivs = [Idot,      # list of dy/dt=f functions,
			  ((gamma*omega_m*np.sin(omega_m*t) - R)/(L0+gamma*np.cos(omega_m*t)))*Idot - I/(C*(L0+gamma*np.cos(omega_m*t)))+np.heaviside(t-t0,V0)]
    return derivs

# Parameters
L0 = 55.00e-12
R = 5.5e-6
C = 12.79e-12
gamma = 0.02
omega_m = 2*pi*(20+0.5852332270276567)*1e3
t0 = 0.01e-5
V0 = 1e-9

# Initial values
I0 = 0
Idot0 = 0.0

# Bundle parameters for ODE solver
params = [L0, R, C, gamma, omega_m,t0, V0]

# Bundle initial conditions for ODE solver
y0 = [I0,Idot0]

# Make time array for solution
tStop = 3e-5
tInc = 1/(8*6e9)
t = np.arange(0., tStop, tInc)

# Call the ODE solver
psoln = odeint(f, y0, t, args=(params,))

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(t, psoln[:,0])
ax1.set_xlabel('time')
ax1.set_ylabel('I')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(t, psoln[:,1])
ax2.set_xlabel('time')
ax2.set_ylabel('Idot')

# Plot omega as a function of time
ax3 = fig.add_subplot(313)
ax3.plot(psoln[:,0], psoln[:,1],'.',ms=1)
ax3.set_xlabel('I')
ax3.set_ylabel('Idot')

plt.tight_layout()
plt.show()
