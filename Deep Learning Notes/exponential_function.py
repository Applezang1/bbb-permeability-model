import numpy as np
import matplotlib.pyplot as plt

'''Exponential Functions'''
# Define an array of x values from -5 to 5 with increments of 0.01
x = np.arange(-5.0,5.0, 0.01)
y = np.exp(x)

# Plot the exponential function
fig, ax = plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([0,100]);ax.set_xlim([-5,5])
ax.set_xlabel('x'); ax.set_ylabel('exp[x]')
plt.show() 

# Define an exponential function
def exponential_function(x): 
   y = np.exp(x)
   return y 

# Compute the exponential function
print(f'The output of the exponential function when x = 0 is {exponential_function(0)}')
print(f'The output of the exponential function when x = 1 is {exponential_function(1)}')
print(f'The output of the exponential function as x approaches infinity is {exponential_function(np.inf)}')
print(f'The output of the exponential function as x approaches negative infinity is {exponential_function(-np.inf)}')