import numpy as np
import matplotlib.pyplot as plt

# Define the model function 
def true_function(x):
    y = np.exp(np.sin(x*(2*3.1413)))
    return y

# Generate some data points with or without noise
def generate_data(n_data, sigma_y=0.3):
    # Generate x values from [0, 1]
    x = np.ones(n_data)
    for i in range(n_data):
        x[i] = np.random.uniform(i/n_data, (i+1)/n_data, 1)

    # Evaluate the output value using the true function
    # Add noise to the output values
    y = np.ones(n_data)
    for i in range(n_data):
        y[i] = true_function(x[i])
        y[i] += np.random.normal(0, sigma_y, 1)
    return x,y

# Define a function to plot the fitted function, together with uncertainty used to generate points
def plot_function(x_func, y_func, x_data=None,y_data=None, x_model = None, y_model =None, sigma_func = None, sigma_model=None):

    fig,ax = plt.subplots()
    ax.plot(x_func, y_func, 'k-')
    if sigma_func is not None:
      ax.fill_between(x_func, y_func-2*sigma_func, y_func+2*sigma_func, color='lightgray')

    if x_data is not None:
        ax.plot(x_data, y_data, 'o', color='#d18362')

    if x_model is not None:
        ax.plot(x_model, y_model, '-', color='#7fe7de')

    if sigma_model is not None:
      ax.fill_between(x_model, y_model-2*sigma_model, y_model+2*sigma_model, color='lightgray')

    ax.set_xlim(0,1)
    ax.set_xlabel('Input, x')
    ax.set_ylabel('Output, y')
    plt.show()
     
# Compute the true function
x_func = np.linspace(0, 1.0, 100)
y_func = true_function(x_func)

# Generate data points with noise
sigma_func = 0.3
n_data = 15
x_data,y_data = generate_data(n_data, sigma_func)

# Plot the function along with the data points and their uncertainty
plot_function(x_func, y_func, x_data, y_data, sigma_func=sigma_func)

# Define neural network model
def network(x, beta, omega):
    # Retrieve number of hidden units
    n_hidden = omega.shape[0]

    y = np.zeros_like(x)
    for c_hidden in range(n_hidden):
        # Evaluate activations based on shifted lines 
        line_vals =  x  - c_hidden/n_hidden
        h =  line_vals * (line_vals > 0)
        # Compute the weighted activations and sum to find the output value
        y = y + omega[c_hidden] * h

    # Add bias, beta (parameters)
    y = y + beta

    return y

# Define a function that computes the value of each hidden unit in a neural network model
def compute_H(x_data, n_hidden):
  psi1 = np.ones((n_hidden+1,1))
  psi0 = np.linspace(0.0, 1.0, num=n_hidden, endpoint=False) * -1

  n_data = x_data.size
  # Compute the hidden variables
  H = np.ones((n_hidden+1, n_data))
  for i in range(n_hidden):
    for j in range(n_data):
      # Compute preactivation
      H[i,j] = psi1[i] * x_data[j]+psi0[i]
      # Apply ReLU to get activation
      if H[i,j] < 0:
        H[i,j] = 0

  return H

# Define a function that returns the covariance and mean matrix of the Bayesian linear regression distribution
def compute_param_mean_covar(x_data, y_data, n_hidden, sigma_sq, sigma_p_sq):
  # Retrieve the matrix containing the hidden variables
  H = compute_H(x_data, n_hidden) 

  # Compute the covariance matrix
  phi_covar = np.linalg.inv((1/sigma_sq)*np.matmul(H, H.T) + (1/sigma_p_sq)*np.identity(H.shape[0]))

  # Compute the mean matrix
  phi_mean = (1/sigma_sq)*np.matmul(phi_covar, np.matmul(H, y_data))

  return phi_mean, phi_covar

# Define parameters for neural network model
n_hidden = 5
sigma_sq = sigma_func * sigma_func
sigma_p_sq = 1000

# Compute the mean and covariance matrix
phi_mean, phi_covar = compute_param_mean_covar(x_data, y_data, n_hidden, sigma_sq, sigma_p_sq)

# Plot the mean model
x_model = x_func
y_model_mean = network(x_model, phi_mean[-1], phi_mean[0:n_hidden])
plot_function(x_func, y_func, x_data, y_data, x_model, y_model_mean)

# Select two samples from the mean and covariance matrix to use for parameter values
samples = np.random.multivariate_normal(phi_mean, phi_covar, size=2)
phi_sample1 = samples[0]
phi_sample2 = samples[1]

# Compute the network for the two chosen sets of parameters
y_model_sample1 = network(x_model, phi_sample1[-1], phi_sample1[0:n_hidden])
y_model_sample2 = network(x_model, phi_sample2[-1], phi_sample2[0:n_hidden])

# Plot the two models
plot_function(x_func, y_func, x_data, y_data, x_model, y_model_sample1)
plot_function(x_func, y_func, x_data, y_data, x_model, y_model_sample2)

# Predict mean and variance of the predicted output value from the input value
def inference(x_star, x_data, y_data, sigma_sq, sigma_p_sq, n_hidden):
  # Compute hidden variables
  h_star = compute_H(x_star, n_hidden)
  H = compute_H(x_data, n_hidden)

  # Compute mean and variance of the predicted output value
  h_star_flat = h_star.reshape(-1)  
  main_term = np.linalg.inv((1/sigma_sq)*np.matmul(H, H.T) + (1/sigma_p_sq)*np.identity(H.shape[0]))
  y_star_mean = (1/sigma_sq)*np.matmul(np.hstack([h_star_flat]), np.matmul(main_term, np.matmul(H, y_data)))
  y_star_var =  np.matmul(np.hstack([h_star_flat]), np.matmul(main_term, np.hstack([h_star_flat])))

  return y_star_mean, y_star_var

# Define array structure to store the model output
x_model = x_func
y_model = np.zeros_like(x_model)
y_model_std = np.zeros_like(x_model)

# Compute the mean and variance of the predicted output from each input value of the model
for c_model in range(len(x_model)):
  y_star_mean, y_star_var = inference(x_model[c_model]*np.ones((1,1)), x_data, y_data, sigma_sq, sigma_p_sq, n_hidden)
  y_model[c_model] = y_star_mean
  y_model_std[c_model] = np.sqrt(y_star_var)
  
# Plot the model
plot_function(x_func, y_func, x_data, y_data, x_model, y_model, sigma_model=y_model_std)