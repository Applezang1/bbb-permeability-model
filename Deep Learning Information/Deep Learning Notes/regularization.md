# Chapter 9: Regularization 
## Regularization 
**<ins>Regularization</ins>**: Regularization is the act of improving the generalization of the model so that there is less divergence between the accuracy of the training data to the test data 

### Explicit Regularization: 
<ins>Explicit Regularization</ins> is when a regularization term is added to the loss function, which takes larger values and increases the loss of a function when certain values of parameters are less preferred. 

The regularization term is identified by:

    - g[theta]: the regularization function that returns larger values when parameters are less preferred 

    - Lambda: scalar term, controls how much of an effect the regularization function has on the loss function 

<ins>Note</ins>: This can also be constructed from the maximum likelihood criterion, where the regularization term is the model’s assumptions on the prior distribution of the parameters (maximum a posteriori/MAP criterion). Therefore, it already has an “assumption” of what the parameters should be before computing the maximum likelihood criterion. 

### L2 Regularization 
<ins>L2 regularization</ins> adds a regularization term called the L2 norm into the sum of squares loss function. L2 regularization applies to the weights, which is why it’s also named the weight decay term 

**<ins>Effects of L2 Regularization</ins>**

    - Favors smaller weights, causing smoother curves but less accurate to the training data 

    - This fixes the problems of overfitting and allows it to generalize better to the test set data 

### Implicit Regularization 
<ins>Implicit Regularization</ins> is the statement that even without adding a regularization term, methods such as the gradient descent or stochastic gradient descent prefer some solutions over others 

#### Implicit Regularization in Gradient Descent
The update of the terms in gradient descent causes an implicit regularization term to be added to the overall loss function 

**<ins>Effects of Implicit Regularization Term</ins>**: 

    - The implicit regularization term is dependent on the computed gradient, where bigger gradients are less favorable than smaller gradients 

    - Because gradient size is dependent on the value of the parameters, where higher parameter values equal bigger gradients, gradient descent favors smaller parameter values

#### Implicit Regularization in Stochastic Gradient Descent 
Similarly, the update of the terms in stochastic gradient descent causes another implicit regularization term to be added to the overall loss function 

**<ins>Effects of Implicit Regularization Term</ins>**: 

    - The implicit regularization term is dependent on the gradient of the batches of data and on the gradient of the overall data 

    - The addition of the implicit regularization term causes the stochastic gradient descent to favor solutions where the gradient of the overall data matches up with the gradients of each particular batch of data (batch variance is small) 

### Heuristics to improve Regularization
#### Early Stopping
**<ins>Early Stopping</ins>**: Early stopping is the act of stopping the training of a neural network model before it has completely matched up with the training set graph. 

<ins>Effects of Early Stopping</ins>: 

    - Early stopping helps prevent overfitting and prevents the model from learning all the random fluctuations and noise of the training data 

    - This allows the neural network model to better generalize predictions for the test data set 

#### Ensembling
**<ins>Ensembling</ins>**:** Ensembling is the act of building and training several models (an ensemble) and averaging their predictions  

<ins>Methods to Train Ensembles</ins>: 

    - Use different initialization values for parameters for each model 

    - Using different training data to train different models each (bootstrap aggregating/bagging) 

#### Dropout
**<ins>Dropout</ins>**: Dropout is the act of making a random subset of hidden units zero for every single gradient descent. 

<ins>Effects of Dropout</ins>: 

    - Dropout prevents a neural network from becoming too dependent on a hidden unit for an output 

    - Because of this behavior, dropout encourages a smaller weight magnitude 
        
    - Note: When running the model after dropout, we need to use the weight scaling inference rule (multiplying the weights by one minus the dropout probability) to compensate for the fact that the neural network model now has more hidden units. In addition, we can also have different dropouts for each run and average the output values (Monte Carlo dropout) 

#### Applying Noise
**<ins>Applying Noise</ins>**: Noise can also be added to the input, weight, or label output data.  

<ins>Effects of Applying Noise</ins>: 

    - Applying noise to the input, weight, or label output data prevents overfitting and improves generalization 

#### Bayesian Inference
**<ins>Bayesian Inference</ins>**: Instead of directly choosing one set of parameter values to use to make predictions, Bayesian inference describes a probability distribution over multiple sets of parameter values 

    - This improves the generalization of the neural network model and captures the model’s uncertainty in the chosen parameter set 

#### Transfer Learning
**<ins>Transfer Learning</ins>**: Transfer learning is the act of using a pre-trained neural network model and fine-tuning or adjusting the model to perform a similar task 

#### Multi-Task Learning
**<ins>Multi-Task Learning</ins>**: Multi-task learning is the act of using a model to compute multiple different types of outputs from a single input. This can improve the model’s performance and help with generalization 

#### Self-Supervised Learning
**<ins>Self-Supervised Learning</ins>**: Self-supervised learning is the act of using a model to generate labels for input data or determine relationships between different labels. 

There are two different types of self-supervised learning: 

<ins>Generative Self-Supervised Learning</ins>: This type of self-supervised learning makes a model predict the label of an input data point 

<ins>Contrastive Self-Supervised Learning</ins>: This type of self-supervised learning makes a model connect and identify relationships between different labels 

Overall, it prevents overfitting and improves the generalization of data because it focuses on the general patterns and relationships of the data 

#### Augmentation
**<ins>Augmentation</ins>**: Augmentation is the act of expanding the dataset by transforming the input data so that more input data is created, but still has the same label/output. Augmentation improves generalization because it exposes the model to a wider range of data. 
