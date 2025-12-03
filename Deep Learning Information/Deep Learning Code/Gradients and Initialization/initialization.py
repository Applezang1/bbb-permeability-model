import numpy as np
import matplotlib.pyplot as plt

def init_params(K, D, sigma_sq_omega):
  # Set seed in order to consistently obtain the same random number
  np.random.seed(0)

  # Input layer
  D_i = 1
  # Output layer
  D_o = 1

  # Construct an empty list for the weights and biases
  all_weights = [None] * (K+1)
  all_biases = [None] * (K+1)

  # Construct an array structure and intialize the weights and biases of the input and output layers. Add varaince to the weights
  all_weights[0] = np.random.normal(size=(D, D_i))*np.sqrt(sigma_sq_omega)
  all_weights[-1] = np.random.normal(size=(D_o, D)) * np.sqrt(sigma_sq_omega)
  all_biases[0] = np.zeros((D,1))
  all_biases[-1]= np.zeros((D_o,1))

  # Construct intermediate hidden layers
  for layer in range(1,K):
    all_weights[layer] = np.random.normal(size=(D,D))*np.sqrt(sigma_sq_omega)
    all_biases[layer] = np.zeros((D,1))

  return all_weights, all_biases
     
# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define a function to compute a neural network
def compute_network_output(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights)-1

  # Store the pre-activations (all_f) and the activations (all_h)
  all_f = [None] * (K+1)
  all_h = [None] * (K+1)

  all_h[0] = net_input

  # Calculate the pre-activation and activation for each hidden layer
  for layer in range(K):
      # Compute the pre-activation function
      all_f[layer] = all_biases[layer] + np.matmul(all_weights[layer], all_h[layer])
      # Compute the activation function
      all_h[layer+1] = ReLU(all_f[layer])

  # Compute the output of the neural network from the last hidden layer
  all_f[K] = all_biases[K] + np.matmul(all_weights[K], all_h[K])

  # Retrieve the output
  net_output = all_f[K]

  return net_output, all_f, all_h

# Define the number of layers
K = 5
# Define the number of neurons per layer
D = 8
# Define the number of input layers
D_i = 1
# Define the number of output layer
D_o = 1

# Set variance of initial weights to 1
sigma_sq_omega = 1.0

# Initialize parameters
all_weights, all_biases = init_params(K,D,sigma_sq_omega)

# Compute the neural network output from the defined parameters
n_data = 1000
data_in = np.random.normal(size=(1,n_data))
net_output, all_f, all_h = compute_network_output(data_in, all_weights, all_biases)

# Compute the standard deviation of the hidden units in each hidden layer
for layer in range(1,K+1):
  print("Layer %d, std of hidden units = %3.3f"%(layer, np.std(all_h[layer])))

# Define the least squares loss function
def least_squares_loss(net_output, y):
  return np.sum((net_output-y) * (net_output-y))

# Define the derivative of the loss function with respect to the output of the neural network
def d_loss_d_output(net_output, y):
    return 2*(net_output -y)

# Define an indicator function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>=0] = 1
  x_in[x_in<0] = 0
  return x_in

# Define a function for the main backward pass
def backward_pass(all_weights, all_biases, all_f, all_h, y):
  # Retrieve number of layers
  K = len(all_weights) - 1

  # Store the derivatives dl_dweights and dl_dbiases in lists
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  # Store the derivatives of the loss with respect to the activation and preactivations in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)

  # Compute derivatives of the loss with respect to the network output
  all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y))

  # Compute the backward pass
  for layer in range(K,-1,-1):
    # Calculate the derivatives of the loss with respect to the biases at layer
    all_dl_dbiases[layer] = np.array(all_dl_df[layer])
    # Calculate the derivatives of the loss with respect to the weights at layer
    all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].transpose())

    # Calculate the derivatives of the loss with respect to the activations
    all_dl_dh[layer] = np.matmul(all_weights[layer].transpose(), all_dl_df[layer])
   
    if layer > 0:
       # Calculate the derivatives of the loss with respect to the pre-activation f
      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer]

  return all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df

# Define the number of layers
K = 5
# Define the number of neurons per layer
D = 8
# Define input layer
D_i = 1
# Define output layer
D_o = 1

# Set variance of initial weights to 1
sigma_sq_omega = 1.0

# Initialize parameters
all_weights, all_biases = init_params(K,D,sigma_sq_omega)

# Initialize array structure to store the gradients
n_data = 100
aggregate_dl_df = [None] * (K+1)
for layer in range(1,K):
  # Store the gradients for every data point in the 3D array
  aggregate_dl_df[layer] = np.zeros((D,n_data))


# Compute the derivatives of the parameters for each data point separately
for c_data in range(n_data):
  data_in = np.random.normal(size=(1,1))
  y = np.zeros((1,1))
  net_output, all_f, all_h = compute_network_output(data_in, all_weights, all_biases)
  all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df = backward_pass(all_weights, all_biases, all_f, all_h, y)
  for layer in range(1,K):
    aggregate_dl_df[layer][:,c_data] = np.squeeze(all_dl_df[layer])

# Print the standard deviation of the derivative of the loss function with respect to the activation function
for layer in reversed(range(1,K)):
  print("Layer %d, std of dl_dh = %3.3f"%(layer, np.std(aggregate_dl_df[layer].ravel())))
