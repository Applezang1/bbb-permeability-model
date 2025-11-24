# Chapter 6: Fitting Models 

## Key Words and Concepts 

**<ins>Epoch</ins>**: a singular pass through the entire training dataset 

**<ins>Hyperparameter Search</ins>**: a method to train models with different hyperparameters in order to choose the best combination of hyperparameters

## Fitting Models 
Fitting a neural network model is when we compute the loss function using the training set of input/output pairs and optimizing the loss function to find parameters that minimize the loss function

The model minimizes the loss function by adjusting the parameters in a way that the loss function decreases 

## Gradient Descent 
Gradient descent is a method to adjust the parameters to minimize the loss functions 

**<ins>Gradient Descent Methods</ins>**

    - The derivatives of the loss function in respect to each parameter (partial derivatives) are computed. This computes the gradient vector of the function, which points towards the direction of steepest increase. 

    - Multiply each partial derivative by a negative constant and add the computed value back into the parameter to adjust the parameter value. By multiplying the partial derivative by a negative, we are moving opposite of the gradient vector (or the direction of steepest decrease) 

<ins>Problem:</ins> 

    - Gradient descent only works for convex functions, which are functions that only have a single global minimum 

    - For non-convex functions (functions with a single global minimum and numerous local minima), gradient descent doesn’t differentiate between a global minimum and local minima so the neural network can get “trapped” at a local minima. 

    - In addition, saddle points are points where the gradient is approaching 0 but has certain directions where the loss function can be minimized more. It’s impossible to determine whether a point where the gradient approaches 0 is a minimum or a saddle point.
 
## Stochastic Gradient Descent 
Stochastic gradient descent is a method to adjust the parameter to minimize the loss functions, but the parameters are adjusted so that the magnitude of decrease is varied and not necessarily the steepest. Stochastic gradient descent can also move uphill, allowing for the possibility of moving from an area of one local minimum to the next. 

**<ins>Stochastic Gradient Descent Methods</ins>**

    - The gradient for a random subset of training data is chosen to be computed. The random subset of data chosen is called the minibatch or batch. If the random subset of data is as large as the entire dataset, it is known as a full-batch gradient descent.  

    - We multiply a negative constant to the computed gradient and add the change back into each parameter of the model. 

<ins>Note</ins>: Instead of changing the subset of training data to compute the gradient for, the stochastic gradient descent might instead change the loss function for each instance and compute the gradient for that loss function. This has the same effect as the stochastic gradient descent defined above. 

**<ins>Advantages of Stochastic Gradient Descent</ins>**

    - Even if stochastic gradient descent only computes the gradient for a particular subset of data, the overall function still moves down the gradient. 

    - Less computation power to compute a stochastic gradient descent, since there’s less training examples chosen per instance 

    - The stochastic gradient descent can escape local minimas through the variation in direction for different subsets of training data 

    - The stochastic gradient descent is less likely to get stuck at saddle points 

## Momentum 
**<ins>Momentum</ins>**: a weighted value whose value depends on the current and all past gradients, where the gradients farther back in time have less weight on the current momentum.  

    - Instead of multiplying by the negative gradient of the current gradient, stochastic gradient descent instead multiplies it by the negative of the current momentum.  

**<ins>Advantages of Momentum</ins>**

    - Momentum is added as a method to smoothen trajectory

    - Momentum reduces oscillatory behavior of the trajectory 

    - Momentum helps the model converge to a minimum faster 

## Nesterov Accelerated Momentum: 
A subset of momentum that looks ahead by calculating the momentum of the current time instance and uses that momentum value to compute the momentum value of the next time interval, essentially moving to the next predicted point

 
Even better reduction of oscillatory behavior, smoothening of trajectory, and converging to a minimum than standard momentum 

## Adaptive Momentum Estimation (Adam)
<ins>Problem</ins>

    - For stochastic gradient descent, the distance moved on the gradient depends on the size of the gradient (large gradients = large adjustments, small gradients = small adjustments) 

    - Therefore by normalizing the gradient, the function will descend down the gradient at constant distance, regardless of the size of the gradient. 

**<ins>Adaptive momentum estimation</ins>** uses the solution to the problem and adds momentum, changing the normalizing of the gradient 

This increases the accuracy and efficiency of the model because its movements depend on the history of past movements from different past gradients as well as its magnitudes. 
