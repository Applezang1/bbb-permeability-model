import numpy as np

# Define function with 8 parameters
def fn(x, beta0, beta1, beta2, beta3, omega0, omega1, omega2, omega3):
  return beta3+omega3 * np.cos(beta2 + omega2 * np.exp(beta1 + omega1 * np.sin(beta0 + omega0 * x)))

# Define a loss function
def loss(x, y, beta0, beta1, beta2, beta3, omega0, omega1, omega2, omega3):
  diff = fn(x, beta0, beta1, beta2, beta3, omega0, omega1, omega2, omega3) - y
  return diff * diff

# Define parameters
beta0 = 1.0; beta1 = 2.0; beta2 = -3.0; beta3 = 0.4
omega0 = 0.1; omega1 = -0.4; omega2 = 2.0; omega3 = 3.0
x = 2.3; y = 2.0

# Compute least squares loss function with defined parameters
l_i_func = loss(x,y,beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3)
print('l_i=%3.3f'%l_i_func)
     
# Define pre activation and activation function
f0 = beta0 + omega0*x
h1 = np.sin(f0)
f1 = beta1 + omega1*h1
h2 = np.exp(f1)
f2 = beta2 + omega2*h2
h3 = np.cos(f2)
f3 = beta3 + omega3*h3

# Define least squares loss function
l_i = (f3 - y)**2

# Ensure that the pre activation and activatin functions were computed properly
print("f0: true value = %3.3f, your value = %3.3f"%(1.230, f0))
print("h1: true value = %3.3f, your value = %3.3f"%(0.942, h1))
print("f1: true value = %3.3f, your value = %3.3f"%(1.623, f1))
print("h2: true value = %3.3f, your value = %3.3f"%(5.068, h2))
print("f2: true value = %3.3f, your value = %3.3f"%(7.137, f2))
print("h3: true value = %3.3f, your value = %3.3f"%(0.657, h3))
print("f3: true value = %3.3f, your value = %3.3f"%(2.372, f3))
print("l_i original = %3.3f, l_i from forward pass = %3.3f"%(l_i_func, l_i))

# Define derivative of the loss function in respect to the pre activation and activation function 
# Backward Pass #1
dldf3 = 2* (f3 - y)
dldh3 = omega3 * dldf3
dldf2 = dldh3 * -np.sin(f2)
dldh2 = dldf2 * omega2
dldf1 = dldh2 * np.exp(f1)
dldh1 = dldf1 * omega1
dldf0 = dldh1 * np.cos(f0)

# Ensure that Backward Pass #1 was computed properly
print("dldf3: true value = %3.3f, your value = %3.3f"%(0.745, dldf3))
print("dldh3: true value = %3.3f, your value = %3.3f"%(2.234, dldh3))
print("dldf2: true value = %3.3f, your value = %3.3f"%(-1.683, dldf2))
print("dldh2: true value = %3.3f, your value = %3.3f"%(-3.366, dldh2))
print("dldf1: true value = %3.3f, your value = %3.3f"%(-17.060, dldf1))
print("dldh1: true value = %3.3f, your value = %3.3f"%(6.824, dldh1))
print("dldf0: true value = %3.3f, your value = %3.3f"%(2.281, dldf0))

# Define the derivative of the loss function in respect to the parameters (beta and omega)
# Backward Pass #2
dldbeta3 = dldf3
dldomega3 = dldf3 * h3
dldbeta2 = dldf2
dldomega2 = dldf2 * h2
dldbeta1 = dldf1
dldomega1 = dldf1 * h1
dldbeta0 = dldf0
dldomega0 = dldf0 * x

# Ensure that Backward Pass #2 was computed properly
print('dldbeta3: Your value = %3.3f, True value = %3.3f'%(dldbeta3, 0.745))
print('dldomega3: Your value = %3.3f, True value = %3.3f'%(dldomega3, 0.489))
print('dldbeta2: Your value = %3.3f, True value = %3.3f'%(dldbeta2, -1.683))
print('dldomega2: Your value = %3.3f, True value = %3.3f'%(dldomega2, -8.530))
print('dldbeta1: Your value = %3.3f, True value = %3.3f'%(dldbeta1, -17.060))
print('dldomega1: Your value = %3.3f, True value = %3.3f'%(dldomega1, -16.079))
print('dldbeta0: Your value = %3.3f, True value = %3.3f'%(dldbeta0, 2.281))
     
     