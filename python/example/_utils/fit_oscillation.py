### Regression for damped oscillation

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data
folder = '../forward_simulation'
dt = 0.01
data = np.loadtxt(f"{folder}/vertical_movement.txt")

# Define function for fitting
def func(t, a, b, c, d, e):
    return a * np.exp(-b * t) * np.cos(c * t + d) + e

# Fit curve
t = np.linspace(0, data.shape[0]*dt, data.shape[0])
popt, pcov, infodict, mesg, ier = curve_fit(func, t, data, p0=(0.1, 2, 1, 0, -0.015), full_output=True)

# Plot data and fit
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(t, func(t, *popt), label="Fitted Curve")
ax.plot(t, data, '--', label="Data")
#ax.set_xlim([0, 0.1])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement in z-direction (m)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.grid()
ax.legend()

fig.savefig(f"{folder}/damped_oscillation.png", dpi=300, bbox_inches='tight')
fig.savefig(f"{folder}/damped_oscillation.pdf", bbox_inches='tight')
plt.close()


# Print error between data and fit
print(f"Error = {np.sqrt(np.sum((func(t, *popt) - data)**2)):.4e}")

### Print physical parameters
print("Fitted Parameters:")
print(f"Amplitude \t\t= {popt[0]:.6f}m")
print(f"Alpha \t\t\t= {popt[1]:.4f}")
print(f"Frequency \t\t= {abs(popt[2])/(2*np.pi):.4f}Hz")
print(f"Phase offset \t\t= {popt[3]:.4f}rad")
print(f"Steady State Offset \t= {popt[4]:.4f}m")

# Print natural frequency
print(f"Natural Frequency \t= {np.sqrt(popt[2]**2 + popt[1]**2) / (2*np.pi):6f}Hz")

mass = 1
print(f"Predicted Stiffness = {mass * (popt[2]**2 + popt[1]**2):.4f}N/m")
print(f"Predicted Damping = {2 * mass * popt[1]:.4f}Ns/m")

thickness = 1/11
youngs_modulus = 1e5
spring_constant = youngs_modulus * thickness**2
print(f"Simulated Stiffness = {spring_constant:.4f}N/m")
print(f"Expected Natural Frequency = {np.sqrt(spring_constant / mass) / (2*np.pi):.4f}Hz")


### Save parameters
np.savetxt(f"{folder}/oscillation_parameters.txt", popt)

