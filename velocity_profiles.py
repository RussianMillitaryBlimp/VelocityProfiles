#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:44:39 2022

@author: diego
"""

import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(8,14))
ax1 = plt.subplot(311)
ax2 = plt.subplot(313)
ax3 = plt.subplot(312)

# fluid properties
mu = 1.8e-5
rho = 1.22

# mesh properties
nx = 201
l = 0.05
dx = l / (nx - 1)

# time stepping
nt = int(1e3)
sigma = 0.5
dt = sigma * dx**2 / (mu/rho)

# distance from the wall
x = np.zeros(nx)
for i in range(nx):
    x[i] = min([dx*i,l-dx*i])

dPdx = 0 # pressure drop

initu = 0.1 # initial velocity

mdot = 0.3 # mass flow rate

# initialise arrays
u = initu*np.ones(nx)
turb = np.zeros(nx)
res = np.ones(nt)
un = np.ones(nx) # placeholder array

print("-"*40) 
print("Reynolds ~", round(mdot/mu), "\t| Mass Flow: ", mdot)

u[0] = 0
u[-1] = 0


for n in range(nt):  # iterate through time
    un = u.copy()
    
    for i in range(1, nx - 1):
        divui = ( un[i+1] - 2*un[i] + un[i-1] ) / dx**2
        graui = ( un[i+1] - un[i-1] ) / (2*dx)
        curvedir = np.sign(divui)
        
        # Turbulence is calculated as rho*l**2*uprime**2 prandtl mixing length        
        turb[i] = rho * \
                  (0.41*x[i])**2 * \
                  (curvedir*abs(graui**2)) # turbulence follows curvature
        
        u[i] = un[i] + (dt/rho)*(mu*(divui) \
                                 - dPdx \
                                 - turb[i] \
                                )

    # correct mass flow rate
    res[n] = np.average(u)
    md = rho*l*res[n]
    
    if mdot >= 1:
        u = u - u*(md-mdot)/mdot
    elif mdot == 0:
        pass
    elif mdot < 1:
        u = u - u*(md-mdot)
        
print("-"*40)    
print("Final Mass Flow: ", rho*l*np.average(u))
print("-"*40) 

ax1.plot(np.linspace(0, l, nx), u);
ax1.set_ylabel("u [m/s]")
ax3.plot(np.linspace(0, l, nx), -turb);
ax3.set_ylabel("turb")
ax2.plot(range(nt), res);
ax2.set_ylabel("residual")
ax2.set_xscale("log")

