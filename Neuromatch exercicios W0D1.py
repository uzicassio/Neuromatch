#%%
"""
Primeiro script Cassio

"""
t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere

print(t_max, dt, tau, el, vr, vth, r, i_mean)


#%%
"""
Segundo script

"""
# Imports

import numpy as np
# Loop for 10 steps, variable 'step' takes values from 0 to 9
for step in range(10):

  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2* np.pi)/0.01))

  # Print value of i
  print(i)
#%%
"""
terceiro script

x = 3.14159265e-1
print(f'{x:.3f}')
--> 0.314

print(f'{x:.4e}')

--> 3.1416e-01
"""

# Initialize step_end
step_end = 10

# Loop for step_end steps
for step in range(step_end):

  # Compute value of t
  t = step * dt

  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Print value of t and i
  print(f'{t:.3f}', f'{i:.4e}')
#%%
"""
quarto script

for step in [0, 1, 2]:
  print(step)

for step in range(3):
  print(step)

start = 0
end = 3
stepsize = 1

for step in range(start, end, stepsize):
  print(step)

"""
t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere



# Initialize step_end and v0
step_end = 11
v = el

# Loop for step_end steps
for step in range(step_end):
  if step==0:
      dt=0
  else if step==1:
      dt= 1e-3
  t = step * dt
  
  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Compute v
  v = v + dt/tau * (el - v +r*i)

  # Print value of t and v
  print(f"{t:.3f} {v:.4e}")
