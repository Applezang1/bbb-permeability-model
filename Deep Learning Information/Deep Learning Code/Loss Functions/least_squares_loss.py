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
    y = np.matmul(beta_1,np.ones((1,n_data))) + np.matmul(omega_1,h1)
    return y 

# Define a function to define the parameters
def get_parameters():
  beta_0 = np.zeros((3,1));  # formerly theta_x0
  omega_0 = np.zeros((3,1)); # formerly theta_x1
  beta_1 = np.zeros((1,1));  # formerly phi_0
  omega_1 = np.zeros((1,3)); # formerly phi_x

  beta_0[0,0] = 0.3; beta_0[1,0] = -1.0; beta_0[2,0] = -0.5
  omega_0[0,0] = -1.0; omega_0[1,0] = 1.8; omega_0[2,0] = 0.65
  beta_1[0,0] = 0.1
  omega_1[0,0] = -2.0; omega_1[0,1] = -1.0; omega_1[0,2] = 7.0

  return beta_0, omega_0, beta_1, omega_1

# Define a function to plot the data
def plot_univariate_regression(x_model, y_model, x_data = None, y_data = None, sigma_model = None, title= None):
  # Format input and output data to 1D arrays
  x_model = np.squeeze(x_model)
  y_model = np.squeeze(y_model)

  fig, ax = plt.subplots()
  ax.plot(x_model,y_model)
  if sigma_model is not None:
    ax.fill_between(x_model, y_model-2*sigma_model, y_model+2*sigma_model, color='lightgray')
  ax.set_xlabel(r'Input, x'); ax.set_ylabel(r'Output, y')
  ax.set_xlim([0,1]);ax.set_ylim([-1,1])
  ax.set_aspect(0.5)
  if title is not None:
    ax.set_title(title)
  if x_data is not None:
    ax.plot(x_data, y_data, 'ko')
  plt.show()

# 1D training data, input/output pairs
x_train = np.array([0.09291784,0.46809093,0.93089486,0.67612654,0.73441752,0.86847339,\
                   0.49873225,0.51083168,0.18343972,0.99380898,0.27840809,0.38028817,\
                   0.12055708,0.56715537,0.92005746,0.77072270,0.85278176,0.05315950,\
                   0.87168699,0.58858043])
y_train = np.array([-0.25934537,0.18195445,0.651270150,0.13921448,0.09366691,0.30567674,\
                    0.372291170,0.20716968,-0.08131792,0.51187806,0.16943738,0.3994327,\
                    0.019062570,0.55820410,0.452564960,-0.1183121,0.02957665,-1.24354444, \
                    0.248038840,0.26824970])

# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()
sigma = 0.2

# Define input values 
x_model = np.arange(0,1,0.01)

# Compute the shallow neural network
y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)

# Plot the shallow neural network
plot_univariate_regression(x_model, y_model, x_train, y_train, sigma_model = sigma)

'''Gaussian Distribution'''
# Define the Gaussian Distribution
def normal_distribution(y, mu, sigma):
    prob = (1/(np.sqrt(2*np.pi*sigma**2)))*np.exp((-(y-mu)**2)/(2*sigma**2))

    return prob

# Ensure that Gaussian Distribution was correctly defined and calculated
print("Correct answer = %3.3f, Your answer = %3.3f"%(0.119,normal_distribution(1,-1,2.3)))

# Plot the Gaussian Distribution.
y_gauss = np.arange(-5,5,0.1)
mu = 0; sigma = 1.0
gauss_prob = normal_distribution(y_gauss, mu, sigma)
fig, ax = plt.subplots()
ax.plot(y_gauss, gauss_prob)
ax.set_xlabel(r'Input, x'); ax.set_ylabel(r'Probability y')
ax.set_xlim([-5,5]);ax.set_ylim([0,1.0])
plt.show()

'''Likelihood Function (Gaussian Distribution)'''
# Define the likelihood function for the Gaussian Distribution
def compute_likelihood(y_train, mu, sigma):
  likelihood = np.prod((1/(np.sqrt(2*np.pi*sigma**2)))*np.exp((-(y_train-mu)**2)/(2*sigma**2)))

  return likelihood

# Compute the likelihood function for a homoscedastic (constant variance/sigma model)
# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()
sigma = 0.2 # Variance

# Compute the mean of the Gaussian using a shallow neural network
mu_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

# Compute the likelihood function of the Gaussian Distribution
likelihood = compute_likelihood(y_train, mu_pred, sigma)

# Ensure that the likelihood function was correctly calculated
print("Correct answer = %9.9f, Your answer = %9.9f"%(0.000010624,likelihood))

'''Negative Log Likelihood Function (Gaussian Distribution)'''
# Define the negative log likelihood function for the Gaussian Distribution
def compute_negative_log_likelihood(y_train, mu, sigma):
  nll = -np.sum(np.log((1/(np.sqrt(2*np.pi*sigma**2)))*np.exp((-(y_train-mu)**2)/(2*sigma**2))))

  return nll

# Compute the likelihood function for a homoscedastic (constant variance/sigma model)
# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()
sigma = 0.2 # Variance

# Compute the mean of the Gaussian using the shallow neural network
mu_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

# Compute the negative log likelihood function of the Gaussian Distribution
nll = compute_negative_log_likelihood(y_train, mu_pred, sigma)

# Ensure that the negative log likelihood function was correctly calculated
print("Correct answer = %9.9f, Your answer = %9.9f"%(11.452419564,nll))

'''Sum of Squares (Gaussian Distribution)'''
# Define the function which returns the squared distance between the observed data (y_train) and the predicted data (y_pred)
def compute_sum_of_squares(y_train, y_pred):
  sum_of_squares = np.sum((y_train - y_pred)**2)

  return sum_of_squares
     
# Test whether the sum of squares function was defined correctly
# Define Parameters
beta_0, omega_0, beta_1, omega_1 = get_parameters()

# Compute the mean of the Gaussian using the shallow neural network
y_pred = mu_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

# Compute the sum of squares between the actual and predicted output value
sum_of_squares = compute_sum_of_squares(y_train, y_pred)

# Ensure that the sum of squares function was correctly calculated
print("Correct answer = %9.9f, Your answer = %9.9f"%(2.020992572,sum_of_squares))

'''Minimize the Loss Function by optimizing beta_1'''
# Define Parameters
beta_1_vals = np.arange(0,1.0,0.01)
beta_0, omega_0, beta_1, omega_1 = get_parameters()
sigma = 0.2

# Define array structure for likelihood, negative log likelihood, and sum of squares
likelihoods = np.zeros_like(beta_1_vals)
nlls = np.zeros_like(beta_1_vals)
sum_squares = np.zeros_like(beta_1_vals)

# Compute the likelihood, negative log likelihood, and sum of squares for each beta_1_vals value
for count in range(len(beta_1_vals)):
  # Set the value for the parameter
  beta_1[0,0] = beta_1_vals[count]

  # Run the network with new parameters
  mu_pred = y_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

  # Compute and store the likelihood, negative log likelihood, and sum of squares
  likelihoods[count] = compute_likelihood(y_train, mu_pred, sigma)
  nlls[count] = compute_negative_log_likelihood(y_train, mu_pred, sigma)
  sum_squares[count] = compute_sum_of_squares(y_train, y_pred)

  # Compute and plot the model for every 20th parameter setting
  if count % 20 == 0:
    # Run the model to get values to plot and plot it.
    y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
    plot_univariate_regression(x_model, y_model, x_train, y_train, sigma_model = sigma, title="beta1=%3.3f"%(beta_1[0,0]))

# Plot the likelihood, negative log likelihood, and least squares for each beta1 value
fig, ax = plt.subplots(1,2)
fig.set_size_inches(10.5, 5.5)
fig.tight_layout(pad=10.0)
likelihood_color = 'tab:red'
nll_color = 'tab:blue'

ax[0].set_xlabel('beta_1[0]')
ax[0].set_ylabel('likelihood', color = likelihood_color)
ax[0].plot(beta_1_vals, likelihoods, color = likelihood_color)
ax[0].tick_params(axis='y', labelcolor=likelihood_color)

ax00 = ax[0].twinx()
ax00.plot(beta_1_vals, nlls, color = nll_color)
ax00.set_ylabel('negative log likelihood', color = nll_color)
ax00.tick_params(axis='y', labelcolor = nll_color)

plt.axvline(x = beta_1_vals[np.argmax(likelihoods)], linestyle='dotted')

ax[1].plot(beta_1_vals, sum_squares); ax[1].set_xlabel('beta_1[0]'); ax[1].set_ylabel('sum of squares')
plt.show()

# Print the maximum likelihood, minimum negative log likelihood, and least squares for the best beta_1 value
print("Maximum likelihood = %3.3f, at beta_1=%3.3f"%( (likelihoods[np.argmax(likelihoods)],beta_1_vals[np.argmax(likelihoods)])))
print("Minimum negative log likelihood = %3.3f, at beta_1=%3.3f"%( (nlls[np.argmin(nlls)],beta_1_vals[np.argmin(nlls)])))
print("Least squares = %3.3f, at beta_1=%3.3f"%( (sum_squares[np.argmin(sum_squares)],beta_1_vals[np.argmin(sum_squares)])))

# Plot the best model
beta_1[0,0] = beta_1_vals[np.argmin(sum_squares)]
y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
plot_univariate_regression(x_model, y_model, x_train, y_train, sigma_model = sigma, title="beta1=%3.3f"%(beta_1[0,0]))

'''Minimize the Loss Function by optimizing sigma'''
# Define Parameters
sigma_vals = np.arange(0.1,0.5,0.005)
beta_0, omega_0, beta_1, omega_1 = get_parameters()
beta_1[0,0] = 0.27

# Define array structure for likelihoods, negative log likelihoods and sum of squares
likelihoods = np.zeros_like(sigma_vals)
nlls = np.zeros_like(sigma_vals)
sum_squares = np.zeros_like(sigma_vals)

for count in range(len(sigma_vals)):
  # Set the value for the parameter
  sigma = sigma_vals[count]

  # Compute the network with new parameters
  mu_pred = y_pred = shallow_nn(x_train, beta_0, omega_0, beta_1, omega_1)

  # Compute and store the likelihood, negative log likelihood, and sum squares
  likelihoods[count] = compute_likelihood(y_train, mu_pred, sigma)
  nlls[count] = compute_negative_log_likelihood(y_train, mu_pred, sigma)
  sum_squares[count] = compute_sum_of_squares(y_train, y_pred)

  # Plot the model for every 20th parameter setting
  if count % 20 == 0:
    # Run the model to get values to plot and plot it.
    y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
    plot_univariate_regression(x_model, y_model, x_train, y_train, sigma_model=sigma, title="sigma=%3.3f"%(sigma))

# Plot the likelihood, negative log likelihood, and least squares for each sigma value
fig, ax = plt.subplots(1,2)
fig.set_size_inches(10.5, 5.5)
fig.tight_layout(pad=10.0)
likelihood_color = 'tab:red'
nll_color = 'tab:blue'

ax[0].set_xlabel('sigma')
ax[0].set_ylabel('likelihood', color = likelihood_color)
ax[0].plot(sigma_vals, likelihoods, color = likelihood_color)
ax[0].tick_params(axis='y', labelcolor=likelihood_color)

ax00 = ax[0].twinx()
ax00.plot(sigma_vals, nlls, color = nll_color)
ax00.set_ylabel('negative log likelihood', color = nll_color)
ax00.tick_params(axis='y', labelcolor = nll_color)

plt.axvline(x = sigma_vals[np.argmax(likelihoods)], linestyle='dotted')

ax[1].plot(sigma_vals, sum_squares); ax[1].set_xlabel('sigma'); ax[1].set_ylabel('sum of squares')
plt.show()

# Print the maximum likelihood and minimum negative log likelihood for the best sigma value
print("Maximum likelihood = %3.3f, at sigma=%3.3f"%( (likelihoods[np.argmax(likelihoods)],sigma_vals[np.argmax(likelihoods)])))
print("Minimum negative log likelihood = %3.3f, at sigma=%3.3f"%( (nlls[np.argmin(nlls)],sigma_vals[np.argmin(nlls)])))

# Plot the best model
sigma= sigma_vals[np.argmin(nlls)]
y_model = shallow_nn(x_model, beta_0, omega_0, beta_1, omega_1)
plot_univariate_regression(x_model, y_model, x_train, y_train, sigma_model = sigma, title="beta_1=%3.3f, sigma =%3.3f"%(beta_1[0,0],sigma))
     