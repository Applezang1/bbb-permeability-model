import numpy as np
import matplotlib.pyplot as plt

'''Logarithmic Function'''
# Define an array of x values from -5 to 5 with increments of 0.01
x = np.arange(0.01,5.0, 0.01)
y = np.log(x)

# Plot the logarithmic function
fig, ax = plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([-5,5]);ax.set_xlim([0,5])
ax.set_xlabel('x'); ax.set_ylabel('$\log[x]$')
plt.show()

# Define a logarithmic function
def logarithmic_function(x): 
   y = np.log(x)
   return y 

# Compute the logarithmic function
print(f'The output of the logarithmic function when x = 0 is {logarithmic_function(0)}')
print(f'The output of the logarithmic function when x = 1 is {logarithmic_function(1)}')
print(f'The output of the logarithmic function when x = e is {logarithmic_function(np.e)}')
print(f'The output of the logarithmic function when x = e^3 is {logarithmic_function(np.e**3)}')
print(f'The output of the logarithmic function when x = -1 is {logarithmic_function(-1)}')