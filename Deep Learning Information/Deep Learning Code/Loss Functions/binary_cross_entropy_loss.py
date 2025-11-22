import numpy as np
import matplotlib.pyplot as plt

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

# Define a shallow neural network
def shallow_nn(x, beta_0, omega_0, beta_1, omega_1):
    # Ensure that input data is (1 x n_data) array
    n_data = x.size
    x = np.reshape(x,(1,n_data))

    # Compute the hidden unit using matrix multiplication for the input array
    h1 = ReLU(np.matmul(beta_0,np.ones((1,n_data))) + np.matmul(omega_0,x))

    # Compute the output of the shallow neural network using matrix multiplication for the input array
    model_out = np.matmul(beta_1,np.ones((1,n_data))) + np.matmul(omega_1,h1)
    return model_out

# Define a function to define the parameters
def get_parameters():
  beta_0 = np.zeros((3,1));  # formerly theta_x0
  omega_0 = np.zeros((3,1)); # formerly theta_x1
  beta_1 = np.zeros((1,1));  # formerly phi_0
  omega_1 = np.zeros((1,3)); # formerly phi_x

  beta_0[0,0] = 0.3; beta_0[1,0] = -1.0; beta_0[2,0] = -0.5
  omega_0[0,0] = -1.0; omega_0[1,0] = 1.8; omega_0[2,0] = 0.65
  beta_1[0,0] = 2.6
  omega_1[0,0] = -24.0; omega_1[0,1] = -8.0; omega_1[0,2] = 50.0

  return beta_0, omega_0, beta_1, omega_1

# Define a function to plot the data
def plot_binary_classification(x_model, out_model, lambda_model, x_data = None, y_data = None, title= None):
  # Format the model data to 1D arrays
  x_model = np.squeeze(x_model)
  out_model = np.squeeze(out_model)
  lambda_model = np.squeeze(lambda_model)

  fig, ax = plt.subplots(1,2)
  fig.set_size_inches(7.0, 3.5)
  fig.tight_layout(pad=3.0)
  ax[0].plot(x_model,out_model)
  ax[0].set_xlabel(r'Input, x'); ax[0].set_ylabel(r'Model output')
  ax[0].set_xlim([0,1]);ax[0].set_ylim([-4,4])
  if title is not None:
    ax[0].set_title(title)
  ax[1].plot(x_model,lambda_model)
  ax[1].set_xlabel(r'Input, x'); ax[1].set_ylabel(r'lambda or Pr(y=1|x)')
  ax[1].set_xlim([0,1]);ax[1].set_ylim([-0.05,1.05])
  if title is not None:
    ax[1].set_title(title)
  if x_data is not None:
    ax[1].plot(x_data, y_data, 'ko')
  plt.show()

'''Sigmoid Function'''
# Sigmoid function: maps [-infty,infty] to [0,1]
def sigmoid(model_out):
  sig_model_out = 1 / (1 + np.exp(-model_out))

  return sig_model_out

# 1D training data, input/output pairs
x_train = np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train = np.array([0,1,1,0,0,1,\
                    1,0,0,1,0,1,\
                    0,1,1,0,1,0, \
                    1,1])

# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()

# Define input values 
x_model = np.arange(0,1,0.01)

# Compute the shallow neural network
model_out= shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)

# Pass the output of the shallow neural network through a sigmoid function
lambda_model = sigmoid(model_out)

# Plot the model and how it matches with the training data
plot_binary_classification(x_model, model_out, lambda_model, x_train, y_train)

'''Bernoulli Distribution'''
def bernoulli_distribution(y, lambda_param):
    prob = np.power(1-lambda_param, 1-y)*lambda_param**y

    return prob

# Ensure that Bernoulli Distribution was correctly defined and calculated
print("Correct answer = %3.3f, Your answer = %3.3f"%(0.8,bernoulli_distribution(0,0.2)))
print("Correct answer = %3.3f, Your answer = %3.3f"%(0.2,bernoulli_distribution(1,0.2)))

'''Likelihood Function (Bernoulli Distribution)'''
# Define the likelihood function for the Bernoulli Distribution
def compute_likelihood(y_train, lambda_param):
  likelihood = np.prod(np.power(1-lambda_param, 1-y_train)*lambda_param**y_train)

  return likelihood

# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()

# Compute the Bernoulli parameter lambda using a shallow neural network 
model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

# Pass the output of the shallow neural network through a sigmoid function
lambda_train = sigmoid(model_out)

# Compute the likelihood function of the Bernoulli Distribution
likelihood = compute_likelihood(y_train, lambda_train)

# Ensure that the likelihood function was correctly calculated
print("Correct answer = %9.9f, Your answer = %9.9f"%(0.000070237,likelihood))

'''Negative Log Likelihood Function (Bernoulli Distribution)'''
def compute_negative_log_likelihood(y_train, lambda_param):
  nll = -np.sum(np.log(np.power(1-lambda_param, 1-y_train)*lambda_param**y_train))

  return nll

# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()

# Compute the Bernoulli parameter lambda using a shallow neural network
model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

# Pass the output of the shallow neural network through a sigmoid function
lambda_train = sigmoid(model_out)

# Compute the negative log likelihood function of the Bernoulli Distribution
nll = compute_negative_log_likelihood(y_train, lambda_train)

# Ensure that the negative log likelihood function was correctly calculated
print("Correct answer = %9.9f, Your answer = %9.9f"%(9.563639387,nll))

'''Minimizing the Loss Function by optimizing beta_1'''
# Define a range of values for the parameter (beta_1)
beta_1_vals = np.arange(-2,6.0,0.1)

# Define array structure for likelihoods and negative log likelihoods
likelihoods = np.zeros_like(beta_1_vals)
nlls = np.zeros_like(beta_1_vals)

beta_0, omega_0, beta_1, omega_1 = get_parameters()
for count in range(len(beta_1_vals)):
  # Set the value for the parameter
  beta_1[0,0] = beta_1_vals[count]

  # Compute the network with the new parameters
  model_out = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)
  lambda_train = sigmoid(model_out)

  # Compute and store the likelihood and the negative log likelihood
  likelihoods[count] = compute_likelihood(y_train,lambda_train)
  nlls[count] = compute_negative_log_likelihood(y_train, lambda_train)

  # Plot the model for every 20th parameter setting
  if count % 20 == 0:
    # Run the model to get values to plot and plot it.
    model_out = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
    lambda_model = sigmoid(model_out)
    plot_binary_classification(x_model, model_out, lambda_model, x_train, y_train, title="beta_1[0]=%3.3f"%(beta_1[0,0]))

# Plot the likelihood and negative log likelihood for each beta_1 value
fig, ax = plt.subplots()
fig.tight_layout(pad=5.0)
likelihood_color = 'tab:red'
nll_color = 'tab:blue'

ax.set_xlabel('beta_1[0]')
ax.set_ylabel('likelihood', color = likelihood_color)
ax.plot(beta_1_vals, likelihoods, color = likelihood_color)
ax.tick_params(axis='y', labelcolor=likelihood_color)

ax1 = ax.twinx()
ax1.plot(beta_1_vals, nlls, color = nll_color)
ax1.set_ylabel('negative log likelihood', color = nll_color)
ax1.tick_params(axis='y', labelcolor = nll_color)

plt.axvline(x = beta_1_vals[np.argmax(likelihoods)], linestyle='dotted')

plt.show()

# Print the maximum likelihood and minimum negative log likelihood for the best beta_1 value
print("Maximum likelihood = %f, at beta_1=%3.3f"%( (likelihoods[np.argmax(likelihoods)],beta_1_vals[np.argmax(likelihoods)])))
print("Minimum negative log likelihood = %f, at beta_1=%3.3f"%( (nlls[np.argmin(nlls)],beta_1_vals[np.argmin(nlls)])))

# Plot the best model
beta_1[0,0] = beta_1_vals[np.argmin(nlls)]
model_out = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
lambda_model = sigmoid(model_out)
plot_binary_classification(x_model, model_out, lambda_model, x_train, y_train, title="beta_1[0]=%3.3f"%(beta_1[0,0]))

     