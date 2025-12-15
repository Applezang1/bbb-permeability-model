import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# Define the model function
def true_function(x):
    y = np.exp(np.sin(x*(2*3.1413)))
    return y

# Generate data points with or without noise
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
np.random.seed(1)
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
        # Compute weighted activations and sum to get the output
        y = y + omega[c_hidden] * h
    # Add bias, beta (parameters)
    y = y + beta

    return y

# Define a function that computes the best parameters for the neural network model
def fit_model_closed_form(x,y,n_hidden):
  n_data = len(x)
  A = np.ones((n_data, n_hidden+1))
  for i in range(n_data):
      for j in range(1,n_hidden+1):
          # Compute Preactivation
          A[i,j] = x[i]-(j-1)/n_hidden
          # Apply the ReLU function
          if A[i,j] < 0:
              A[i,j] = 0

  # Add a regularization term to the loss function
  reg_value = 0.00001
  regMat = reg_value * np.identity(n_hidden+1)
  regMat[0,0] = 0

  ATA = np.matmul(np.transpose(A), A) +regMat
  ATAInv = np.linalg.inv(ATA)
  ATAInvAT = np.matmul(ATAInv, np.transpose(A))
  beta_omega = np.matmul(ATAInvAT,y)
  beta = beta_omega[0]
  omega = beta_omega[1:]

  return beta, omega

# Compute the best parameters that results in the minimum of the loss function
beta, omega = fit_model_closed_form(x_data,y_data,n_hidden=14)

# Compute the output using the neural network model
x_model = np.linspace(0,1,100)
y_model = network(x_model, beta, omega)

# Plot the true function and the model
plot_function(x_func, y_func, x_data,y_data, x_model, y_model)

# Compute the mean squared error between the fitted model (cyan) and the true function (black)
mean_sq_error = np.mean((y_model-y_func) * (y_model-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")

# Define the number of models
n_model = 4

# Define an array structure to store the prediction from all the models
all_y_model = np.zeros((n_model, len(y_model)))

# Run the loop for each model
for c_model in range(n_model):
    # Resample data to create a subset of the main dataset for the model
    resampled_indices = np.random.choice(np.arange(0,n_data,1), size=n_data)

    # Extract the resampled x and y data
    x_data_resampled = x_data[resampled_indices]
    y_data_resampled = y_data[resampled_indices]

    # Train the model
    beta, omega = fit_model_closed_form(x_data_resampled,y_data_resampled,n_hidden=14)

    # Compute the model
    y_model_resampled = network(x_model, beta, omega)

    # Store the output
    all_y_model[c_model,:] = y_model_resampled

    # Plot the true function and the model
    plot_function(x_func, y_func, x_data,y_data, x_model, y_model_resampled)

    # Compute the mean squared error between the fitted model (cyan) and the true function wh(black)
    mean_sq_error = np.mean((y_model_resampled-y_func) * (y_model_resampled-y_func))
    print(f"Mean square error = {mean_sq_error:3.3f}")

# Compute the median of the outputs of each neural network model
y_model_median = np.median(all_y_model,axis = 0)

# Plot the true function and the model (fitted through the computed median)
plot_function(x_func, y_func, x_data,y_data, x_model, y_model_median)

# Compute the mean squared error between the fitted model (cyan) and the true function (black)
mean_sq_error = np.mean((y_model_median-y_func) * (y_model_median-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")

# Compute the mean of the outputs of each neural network model
y_model_mean = np.mean(all_y_model, axis = 0)

# Plot the true function and the model (fitted through the computed mean)
plot_function(x_func, y_func, x_data,y_data, x_model, y_model_mean)

# Compute the mean squared error between the fitted model (cyan) and the true function (black)
mean_sq_error = np.mean((y_model_mean-y_func) * (y_model_mean-y_func))
print(f"Mean square error = {mean_sq_error:3.3f}")