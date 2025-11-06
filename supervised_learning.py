import numpy as np 
import matplotlib.pyplot as plt 

# Define Input and Output Data
x = np.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90])
y = np.array([0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6 ])

print(x)
print(y) 

'''1D Linear Regression Model'''
# Define 1D Linear Regression Model
def f(x, phi0, phi1):
  y = phi0 + phi1*x
  return y
     
# Define a function to plot the 1D Linear Regression Model
def plot(x, y, phi0, phi1):
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    plt.xlim([0,2.0])
    plt.ylim([0,2.0])
    ax.set_xlabel('Input, x')
    ax.set_ylabel('Output, y')
    # Draw line
    x_line = np.arange(0,2,0.01)
    y_line = f(x_line, phi0, phi1)
    plt.plot(x_line, y_line,'b-',lw=2)

    plt.show()

# Define the parameters 
# phi0 = y-intercept, phi1 = slope
phi0 = 0.4 ; phi1 = 0.2

# Graph the 1D Linear Regression Model using the input and output data
plot(x,y,phi0,phi1)

'''Least Squares Loss Function'''
# Define the least squares loss function
def compute_loss(x,y,phi0,phi1):
   loss = []
   for i in range(0, x.size, 1): 
      loss_value = (phi0 + phi1*x[i] - y[i])**2 
      loss.append(loss_value) 
   loss = sum(loss)
   return loss

# Compute the loss for the 1D Linear Regression Model
loss = compute_loss(x,y,phi0,phi1)
print(f'Your Loss = {loss:3.2f}, Ground truth =7.07')

'''1D Linear Regression Model with different parameters'''
# Define the parameters with a new set of values
phi0 = 1.60 ; phi1 =-0.8

# Graph the 1D Linear Regression Model using the input and output data and the new parameters
plot(x,y,phi0,phi1)

# Compute the least squares loss function of the Regression Model with the new parameters
loss = compute_loss(x,y,phi0,phi1)
print(f'Your Loss = {loss:3.2f}, Ground truth =10.28')

# Define a more optimal set of parameters
phi0 = 0.75 ; phi1 =0.6

'''1D Linear Regression Model with optimal parameters'''
# Graph the 1D Linear Regression Model using the optimal parameters
plot(x,y,phi0,phi1)

# Compute the least squares loss function of the Regression Model with the optimal parameters
print(f'Your Loss = {compute_loss(x,y,phi0,phi1):3.2f}')

'''Visualize the loss function'''
# Make a 2D grid of possible phi0 and phi1 values
phi0_mesh, phi1_mesh = np.meshgrid(np.arange(0.0,2.0,0.02), np.arange(-1.0,1.0,0.02))

# Make a 2D array for the loss values
all_losses = np.zeros_like(phi1_mesh)

# Run through each 2D combination of phi0, phi1 and compute loss
for indices,temp in np.ndenumerate(phi1_mesh):
    all_losses[indices] = compute_loss(x,y, phi0_mesh[indices], phi1_mesh[indices])

# Plot the loss function as a heatmap
fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(7,7)
levels = 256
ax.contourf(phi0_mesh, phi1_mesh, all_losses ,levels)
levels = 40
ax.contour(phi0_mesh, phi1_mesh, all_losses ,levels, colors=['#80808080'])
ax.set_ylim([1,-1])
ax.set_xlabel(r'Intercept, phi0')
ax.set_ylabel(r'Slope, phi1')

# Plot the position of your best fitting line on the loss function
ax.plot(phi0,phi1,'ro')
plt.show()