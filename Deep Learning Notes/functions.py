import numpy as np 
import matplotlib.pyplot as plt 

'''1D Linear Function'''
# Define a linear function with one input, x 
# beta: y-intercept, omega: slope
def linear_function_1D(x, beta, omega): 
    y = beta + omega*x
    return y

# Define variables
# Define an array of x values from 0 to 10 with increments of 0.01 
x = np.arange(0.0, 10.0, 0.01) 
beta = 10.0; omega = -2.0

# Compute the 1D linear function
y = linear_function_1D(x, beta, omega)

# Plot the 1D linear function
fig, ax = plt.subplots()
ax.plot(x,y,'r-')
ax.set_ylim([0,10]);ax.set_xlim([0,10])
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.show()

'''2D Linear Function'''
# Define a function to draw 2D function
def draw_2D_function(x1_mesh, x2_mesh, y):
    fig, ax = plt.subplots()
    fig.set_size_inches(7,7)
    pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256 ,cmap = 'hot', vmin=-10,vmax=10.0)
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('x1');ax.set_ylabel('x2')
    levels = np.arange(-10,10,1.0)
    ax.contour(x1_mesh, x2_mesh, y, levels, cmap='winter')
    plt.show()

# Define a linear function with two inputs, x1 and x2
def linear_function_2D(x1,x2,beta,omega1,omega2):
  y = beta + omega1*x1 + omega2*x2
  return y

# Define input values by defining a 2D array of x and y points
x1 = np.arange(0.0, 10.0, 0.1)
x2 = np.arange(0.0, 10.0, 0.1)
x1,x2 = np.meshgrid(x1,x2)  

# Define the variables
beta = 0.0; omega1 = 1.0; omega2 = -0.5

# Compute the 2D function for given values of omega1, omega2
y  = linear_function_2D(x1,x2,beta, omega1, omega2)

# Graph the 2D function.
draw_2D_function(x1,x2,y) 

'''3D Linear Function'''
# Define a linear function with three inputs, x1, x2, and x3
def linear_function_3D(x1,x2,x3,beta,omega1,omega2,omega3):
  y = beta + omega1*x1 + omega2*x2 + omega3*x3
  return y

# Define the parameters
beta1 = 0.5; beta2 = 0.2
omega11 =  -1.0 ; omega12 = 0.4; omega13 = -0.3
omega21 =  0.1  ; omega22 = 0.1; omega23 = 1.2

# Define the inputs
x1 = 4 ; x2 =-1; x3 = 2

# Compute using the individual equations
y1 = linear_function_3D(x1,x2,x3,beta1,omega11,omega12,omega13)
y2 = linear_function_3D(x1,x2,x3,beta2,omega21,omega22,omega23)
print("Individual equations")
print('y1 = %3.3f\ny2 = %3.3f'%((y1,y2)))

'''3D Linear Function using Vectors and Matrices'''
# Define vectors and matrices
beta_vec = np.array([[beta1],[beta2]])
omega_mat = np.array([[omega11,omega12,omega13],[omega21,omega22,omega23]])
x_vec = np.array([[x1], [x2], [x3]])

# Compute with vector/matrix form
y_vec = beta_vec+np.matmul(omega_mat, x_vec)
print("Matrix/vector form")
print('y1= %3.3f\ny2 = %3.3f'%((y_vec[0][0],y_vec[1][0])))

'''3D Linear Function with 2 Inputs'''
# Define a linear function with two inputs, x1, x2
def linear_function_3D(x1,x2,beta,omega1,omega2):
  y = beta + omega1*x1 + omega2*x2
  return y 

# Define the parameters
beta1 = 0.5; beta2 = 0.2
omega11 =  -1.0 ; omega12 = 0.4
omega21 =  0.1  ; omega22 = 0.1

# Define the inputs
x1 = 4 ; x2 =-1

# Compute using the individual equations
y1 = linear_function_3D(x1,x2,beta1,omega11,omega12)
y2 = linear_function_3D(x1,x2,beta2,omega21,omega22)
print("Individual equations")
print('y1 = %3.3f\ny2 = %3.3f'%((y1,y2)))

'''3D Linear Function with 2 Inputs using Vectors and Matrices'''
# Define vectors and matrices
beta_vec = np.array([[beta1],[beta2]])
omega_mat = np.array([[omega11,omega12],[omega21,omega22]])
x_vec = np.array([[x1], [x2]])

# Compute with vector/matrix form
y_vec = beta_vec+np.matmul(omega_mat, x_vec)
print("Matrix/vector form")
print('y1= %3.3f\ny2 = %3.3f'%((y_vec[0][0],y_vec[1][0])))


