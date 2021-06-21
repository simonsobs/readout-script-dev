from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

Pi = np.pi


def vectorfield(w, t, p):
     """
     Defines the differential equations for the coupled spring-mass system.

     Arguments:
         w :  vector of the state variables:
                   w = [x1,y1,x2,y2]
         t :  time
         p :  vector of the parameters:
                   p = [m1,m2,k1,k2,L1,L2,b1,b2]
     """
     B,A,I,I2,phi = w
     Rr, Lr, Cr, Cc, L0, g, omega = p

     # Create f = (B',A',I',I2'):
     f = [(-Rr/Lr)*B-(1/(Lr*Cr))*I2+(Rr/Lr)*A+(1/(Lr*Cr))*I,
          (1/(L0+g*np.cos(phi)))*(Rr*B+(1/Cr)*I2-Rr*A-(1/Cr
                )*I-(g*omega*np.sin(phi))*A+(1/Cc)*I),
          A,
          B,
          omega]
     return f

#Initialize Parameters
Rr = 10.0
Lr = 54.9e-12
Cr = 0.1e-6;
Cc = 0.1*1e-6;
L0 = 22.3e-12;
g = (1/100)*L0;
omega = 2*Pi*20e3

#Initial Conditions
B0 = 1e-3
A0 = 2e-3
I0 = 1
I20 = 0.9
phi0 = Pi

# ODE solver parameters
abserr = 1.0e-12
relerr = 1.0e-10
stoptime = 180.0
numpoints = 1000

#Initialize time array
t = np.arange(0,1/(4*20e3),1/(4e9))
#t = np.linspace(0,100,1000)

# Pack up the parameters and initial conditions:
p = [Rr, Lr, Cr, Cc, L0, g, omega]
w0 = [B0,A0,I0,I20,phi0]

# Call the ODE solver.
wsol,info = odeint(vectorfield, w0, t, args=(p,),
    atol=abserr,rtol=relerr,
    full_output = True)

print(wsol)
print(info)

#plt.loglog(t,wsol[:,0],label = 'dI2/dt')
#plt.loglog(t,wsol[:,1],label = 'dI/dt')
plt.loglog(t,wsol[:,2],label = 'I')
#plt.loglog(t,wsol[:,3],label = 'I2')
#plt.loglog(t,wsol[:,4],label = 'Phi')
plt.show()
