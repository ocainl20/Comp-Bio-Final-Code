#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:40:20 2023

@author: laurenocain
"""

# load  packages
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import colors

 

 

# goal: ecological model to show how high interspecific competition affects a population and the impact of disease and other outside factors
# ex:  steep increase in the population of predators
# use lotka-Volterra equations to model behaviors

 

#%% model population with stable coexistence 
# where intraspecific competition is greater than the interspecific competition

 

# Define Function (matrix form)
def gLV_dynamics(t, y, n, mu, alpha):
# y = species (n total)
    dydt = np.zeros(n)

# vectorized form of gLV model
    dydt = y * (mu +np.matmul(alpha, y))

    return dydt

 

# Define Parameters
n = 3


# parameters for prey
a = 0.1 # prey growth rate
b = 0.02 # prey death rate (based on interaction)

 

# parameters for predator
c = 0.5 # predator natural death
d = 0.01 # predator growth rate (based on interaction)

 

# parameters for parasite
e = -.2 # parasite growth rate  # change e from -0.05 to -.5
f = 0.1 # parasite death rate

mu=np.array([a, -c, f])

 


var = -0.1
alpha = np.array([[var, -b, e], [d, var, e], [f, f, var]])
# interaction coefficients, dimensions n x n

 

 

 

# Initial Conditions
y0 = [10, 10, 10]
 # change parasite IC from 10, 6, 2, ...

 

# time span
t = np.linspace(0, 5, 1000)
tspan = [t[0],t[-1]]

 

# numerical integration
ode_sol = solve_ivp(gLV_dynamics, tspan, y0, t_eval = t, args = (n, mu, alpha))

 

# plot dynamics
plt.figure('weewoo', clear = True)

 

plt.plot(ode_sol.t, ode_sol.y[0], 'm-')
plt.plot(ode_sol.t, ode_sol.y[1], 'b-')
plt.plot(ode_sol.t, ode_sol.y[2], 'r-')
plt.xlabel('Time [s]')
plt.ylabel('Population density')
plt.legend(['Lynx', 'Hare', 'Parasite'])

 


#%% # Create Interaction Matrix

 


# parameters for prey
a = 0.1 # prey growth rate
b = -0.02 # prey death rate (based on interaction)

 


# parameters for predator
c = 0.5 # predator natural death
d = 0.01 # predator growth rate (based on interaction)

 

# parameters for parasite
e = -.07 # parasite growth rate  # change e from -0.05 to -.5
f = 0.1 # parasite death rate

 


# Define Parameters
mu_1 = [a, -c, f]
mu_2 = [a, -c, f]
mu_3 = [a, -c, f]
mu_4 = [a, -c, f]

 

# alpha_1 = prey and predator interactions ONLY
alpha_1 = np.array([[var, b, 0], [d, var, 0], [0, 0, var]])

 

# alpha_2 = prey and predator interactions, parasite and predator interactions
alpha_2 = np.array([[var, b, 0], [d, var, e], [0, f, var]])

 

# alpha_3 = prey and predator interactions, parasite and prey interactions
alpha_3 = np.array([[var, b, e], [d, var, 0], [f, 0, var]])

 

# alpha_4 = all interactions
alpha_4 = np.array([[var, -b, e], [d, var, e], [f, f, var]])

 


# List of organisms
orgs = np.array(['PREY','PRED','PARA'])

 

# create plots
[fig, axs] = plt.subplots(2, 2, figsize=[8,7])

 

minval = np.min([alpha_1, alpha_2, alpha_3, alpha_4])
maxval = np.max([alpha_1, alpha_2, alpha_3, alpha_4])

 


axs[0, 0].imshow(alpha_1, cmap ="PiYG", vmin = minval, vmax = maxval)
axs[0, 0].set_title('alpha_1')
axs[0, 0].set_xticks(range(3))
axs[0, 0].set_xticklabels(orgs)
axs[0, 0].set_yticks(range(3))
axs[0, 0].set_yticklabels(orgs)

 


axs[0, 1].imshow(alpha_2, cmap ="PiYG", vmin = minval, vmax = maxval)
axs[0, 1].set_title('alpha_2')
axs[0, 1].set_xticks(range(3))
axs[0, 1].set_xticklabels(orgs)
axs[0, 1].set_yticks(range(3))
axs[0, 1].set_yticklabels(orgs)

 

 

axs[1, 0].imshow(alpha_3, cmap ="PiYG", vmin = minval, vmax = maxval)
axs[1, 0].set_title('alpha_3')
axs[1, 0].set_xticks(range(3))
axs[1, 0].set_xticklabels(orgs)
axs[1, 0].set_yticks(range(3))
axs[1, 0].set_yticklabels(orgs)

 

 

im = axs[1, 1].imshow(alpha_4, cmap ="PiYG", vmin = minval, vmax = maxval)
axs[1, 1].set_title('alpha_4');
axs[1, 1].set_xticks(range(3))
axs[1, 1].set_xticklabels(orgs)
axs[1, 1].set_yticks(range(3))
axs[1, 1].set_yticklabels(orgs);

 

fig.subplots_adjust(right = 0.8)

 

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

 


divnorm = colors.TwoSlopeNorm(vmin = minval, vcenter=0.0, vmax = maxval)

 

ticks = np.arange(minval, maxval, .05)
print(ticks)

 

cbar = fig.colorbar(mappable = None, cax = cbar_ax, cmap="PiYG", norm = divnorm, ticks=ticks)

 

cbar.ax.set_ticks(ticks)

 

plt.show()