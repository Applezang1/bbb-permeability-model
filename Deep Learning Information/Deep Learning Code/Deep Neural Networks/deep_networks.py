import numpy as np
import matplotlib.pyplot as plt

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define a shallow neural network with, one input, one output, and three hidden units
def shallow_1_1_3(x, activation_fn, phi_0,phi_1,phi_2,phi_3, theta_10, theta_11, theta_20, theta_21, theta_30, theta_31):
  
  # Initial lines
  pre_1 = theta_10 + theta_11 * x
  pre_2 = theta_20 + theta_21 * x
  pre_3 = theta_30 + theta_31 * x

  # Activation functions
  act_1 = activation_fn(pre_1)
  act_2 = activation_fn(pre_2)
  act_3 = activation_fn(pre_3)

  # Weight activations
  w_act_1 = phi_1 * act_1
  w_act_2 = phi_2 * act_2
  w_act_3 = phi_3 * act_3

  # Combine weighted activation and add y offset
  y = phi_0 + w_act_1 + w_act_2 + w_act_3

  return y, pre_1, pre_2, pre_3, act_1, act_2, act_3, w_act_1, w_act_2, w_act_3

# Define a function to plot the shallow neural network
def plot_neural(x, y):
  fig, ax = plt.subplots()
  ax.plot(x.T,y.T)
  ax.set_xlabel('Input'); ax.set_ylabel('Output')
  ax.set_xlim([-1,1]);ax.set_ylim([-1,1])
  ax.set_aspect(1.0)
  plt.show()

'''First Shallow Neural Network'''
# Define Parameters
n1_theta_10 = 0.0   ; n1_theta_11 = -1.0
n1_theta_20 = 0     ; n1_theta_21 = 1.0
n1_theta_30 = -0.67 ; n1_theta_31 =  1.0
n1_phi_0 = 1.0; n1_phi_1 = -2.0; n1_phi_2 = -3.0; n1_phi_3 = 9.3

# Define a range of input values
n1_in = np.arange(-1,1,0.01).reshape([1,-1])

# Compute the shallow neural network
n1_out, *_ = shallow_1_1_3(n1_in, ReLU, n1_phi_0, n1_phi_1, n1_phi_2, n1_phi_3, n1_theta_10, n1_theta_11, n1_theta_20, n1_theta_21, n1_theta_30, n1_theta_31)

# Plot the shallow neural network
plot_neural(n1_in, n1_out)

'''First Shallow Neural Network (Matrix Form)'''
# Define Parameter Structure in Matrix Form
beta_0 = np.zeros((3,1))
Omega_0 = np.zeros((3,1))
beta_1 = np.zeros((1,1))
Omega_1 = np.zeros((1,3))

beta_0[0,0] = n1_theta_10
beta_0[1,0] = n1_theta_20 
beta_0[2,0] = n1_theta_30

Omega_0[0,0] = n1_theta_11
Omega_0[1,0] = n1_theta_21 
Omega_0[2,0] = n1_theta_31 

beta_1[0,0] = n1_phi_0

Omega_1[0,0] = n1_phi_1
Omega_1[0,1] = n1_phi_2
Omega_1[0,2] = n1_phi_3

# Make sure that input data matrix has different inputs in its columns
n_data = n1_in.size
n_dim_in = 1
n1_in_mat = np.reshape(n1_in,(n_dim_in,n_data))

# This runs the network for ALL of the inputs, x at once so we can draw graph
h1 = ReLU(beta_0 + np.matmul(Omega_0,n1_in_mat))
n1_out = beta_1 + np.matmul(Omega_1,h1)

# Plot the neural network (Matrix Form)
plot_neural(n1_in, n1_out)

'''Second Neural Network'''
# Define Parameters
n2_theta_10 =  -0.6 ; n2_theta_11 = -1.0
n2_theta_20 =  0.2  ; n2_theta_21 = 1.0
n2_theta_30 =  -0.5  ; n2_theta_31 =  1.0
n2_phi_0 = 0.5; n2_phi_1 = -1.0; n2_phi_2 = -1.5; n2_phi_3 = 2.0

# Compute the second neural network by using the output of the first neural network as the input of the second neural network
n2_out, *_ = \
    shallow_1_1_3(n1_out, ReLU, n2_phi_0, n2_phi_1, n2_phi_2, n2_phi_3, n2_theta_10, n2_theta_11, n2_theta_20, n2_theta_21, n2_theta_30, n2_theta_31)

# Plot the second neural network
plot_neural(n1_in, n2_out)

'''Second Neural Network (Matrix Form)'''
# Define Parameter Structure in Matrix Form
beta_0 = np.zeros((3,1))
Omega_0 = np.zeros((3,1))
beta_1 = np.zeros((3,1))
Omega_1 = np.zeros((3,3))
beta_2 = np.zeros((1,1))
Omega_2 = np.zeros((1,3))

beta_0[0,0] = n1_theta_10
beta_0[1,0] = n1_theta_20 
beta_0[2,0] = n1_theta_30 

Omega_0[0,0] = n1_theta_11
Omega_0[1,0] = n1_theta_21
Omega_0[2,0] = n1_theta_31

beta_1[0,0] = n2_theta_10 + n2_theta_11 * n1_phi_0
beta_1[1,0] = n2_theta_20 + n2_theta_21 * n1_phi_0
beta_1[2,0] = n2_theta_30 + n2_theta_31 * n1_phi_0

Omega_1[0,0] = n2_theta_11 * n1_phi_1 
Omega_1[1,0] = n2_theta_21 * n1_phi_1
Omega_1[2,0] = n2_theta_31 * n1_phi_1

Omega_1[0,1] = n2_theta_11 * n1_phi_2
Omega_1[1,1] = n2_theta_21 * n1_phi_2
Omega_1[2,1] = n2_theta_31 * n1_phi_2

Omega_1[0,2] = n2_theta_11 * n1_phi_3 
Omega_1[1,2] = n2_theta_21 * n1_phi_3
Omega_1[2,2] = n2_theta_31 * n1_phi_3 

beta_2[0,0] = n2_phi_0 

Omega_2[0,0] = n2_phi_1
Omega_2[0,1] = n2_phi_2
Omega_2[0,2] = n2_phi_3

# Make sure that input data matrix has different inputs in its columns
n_data = n1_in.size
n_dim_in = 1
n1_in_mat = np.reshape(n1_in,(n_dim_in,n_data))

# This runs the network for ALL of the inputs, x at once so we can draw graph (hence extra np.ones term)
h1 = ReLU(beta_0 + np.matmul(Omega_0,n1_in_mat))
h2 = ReLU(beta_1 + np.matmul(Omega_1,h1))
n1_out = beta_2 + np.matmul(Omega_2,h2)

# Plot neural network (Matrix Form)
plot_neural(n1_in, n1_out)
     
'''Compute Deep Neural Network with given input, hidden unit, and output sizes'''
# Define Sizes
D_i=4; D_1=5; D_2=2; D_3=4; D_o=1

# Define Input and Parameters (Matrix Form)
n_data = 4
x = np.random.normal(size=(D_i, n_data))
beta_0 = np.random.normal(size=(5,1))
Omega_0 = np.random.normal(size=(5,4))
beta_1 = np.random.normal(size=(2,1))
Omega_1 = np.random.normal(size=(2,5))
beta_2 = np.random.normal(size=(4,1))
Omega_2 = np.random.normal(size=(4,2))
beta_3 = np.random.normal(size=(1,1))
Omega_3 = np.random.normal(size=(1,4))

# Compute the value of hidden units
h1 = ReLU(beta_0 + np.matmul(Omega_0,x))
h2 = ReLU(beta_1 + np.matmul(Omega_1,h1))
h3 = ReLU(beta_2 + np.matmul(Omega_2,h2))
y = beta_3 + np.matmul(Omega_3,h3)

# Check whether the matrix's shape is valid or not
if h1.shape[0] is not D_1 or h1.shape[1] is not n_data:
  print("h1 is wrong shape")
if h2.shape[0] is not D_2 or h1.shape[1] is not n_data:
  print("h2 is wrong shape")
if h3.shape[0] is not D_3 or h1.shape[1] is not n_data:
  print("h3 is wrong shape")
if y.shape[0] is not D_o or h1.shape[1] is not n_data:
  print("Output is wrong shape")

# Print the inputs and outputs
print("Input data points")
print(x)
print ("Output data points")
print(y)