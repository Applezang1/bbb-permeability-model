# Chapter 5: Loss Functions 

## Method for Creating Loss Functions 
<ins>Note</ins>: A model computes the conditional probability distribution of a wide range of possible output values given an input value, and not a single output value. To determine a point estimate using a probability distribution, return the output value that has the highest distribution 

1) Therefore for any model, a parametric distribution with ranges that match the output domain must be chosen. This is known as the likelihood equation which computes the probability for a certain output given an input value. 

2) Then, the maximum likelihood criterion is used to compute the parameter that causes the output to most resemble the defined output distribution  

3) To simplify the maximum likelihood criterion and make calculations easier, take the logarithm of the maximum likelihood criterion (now called log-likelihood criterion) 

4) The log-likelihood criterion is multiplied by the negative one to create the negative log-likelihood criterion, which forms the loss function. The negative log-likelihood criterion finds the parameter that minimizes the divergence from the predicted probability distribution to the actual probability distribution

<ins>Note</ins>: By using the maximum likelihood criterion, we are assuming that 

- The data is identically distributed (each data point is computed using the same probability function)

- The data is independent (one data point doesn’t influence another) 

- Conclusion: This is why we’re able to multiply the probability distributions of different input values of the same parameter (same probability function) 

## Different Loss Functions  
### Univariate Regression 
**<ins>Univariate Regression</ins>**: a model that predicts a single output value (domain: all real numbers) given an input value x 

<ins>Parametric Distribution</ins>: Choose the univariate normal distribution as the parametric distribution for the model, which is defined over all real output values 

<ins>Model Output</ins>: The univariate normal distribution has two parameters (mean, variance). Define the model to predict the value of the mean and treat the variance as a constant. 

- <ins>Note</ins>: By treating the variance as a constant, we are assuming that the output increases relatively the same throughout all input values 

Using the univariate normal distribution and the model output, we are able to compute the negative log-likelihood and simplify it. This results in the least squares loss function

### Heteroscedastic Regression 
The <ins>heteroscedastic regression</ins> uses the same format as the univariate regression but has two different models (one for the mean and another for the variance). This assumes that the variance isn’t constant, making it a suitable loss function for heteroscedastic (nonconstant variance) models

### Binary Classification 
**<ins>Binary Classification</ins>**: an output of one of two discrete values (0, 1) 

<ins>Parametric Distribution</ins>: Choose the Bernoulli distribution as the parametric distribution for the model, which outputs the probability that an output value is 1 and has a domain input of {0, 1} 

<ins>Input Function</ins>: Because the Bernoulli distribution only takes input values on the domain {0, 1}, use the logistic sigmoid function to guarantee that the input value is between 0 and 1 

Compute the negative log-likelihood using the Bernoulli distribution, which takes inputs that are passed through the logistic sigmoid function. The resulting loss function is called the binary cross-entropy loss function. 

### Multiclass Classification 
**<ins>Multiclass Classification</ins>**: assigning an input value to one of many (2+) values (example: predicting which number is drawn) 

<ins>Parametric Distribution</ins>: Choose the categorical distribution, which is defined on all real output values with each parameter determining the probability of each output value 

<ins>Output Function</ins>: Pass the output of the categorical distribution function through the softmax function, which turns the raw outputs into probabilities of a certain output value. 

Compute the negative log-likelihood using the categorical distribution and the softmax function to express the resulting loss function. The resulting loss function is called the multiclass cross-entropy loss. 

## Multiple Outputs 
For a model with multiple outputs (melting + boiling point), we must assume that each of these outputs are independent of each other 

Therefore, we can predict the probability of both of those outputs by multiplying the probability for each of the outputs (this will become the likelihood equation). We can then compute a negative log probability and minimize it to find the loss function. 

In addition, cross-entropy loss can also be used to obtain the format of the negative log-likelihood criterion.

### Cross-Entropy Loss 
**<ins>Cross Entropy Loss</ins>**: finding the parameters that most minimize the loss function (the difference between the predicted output and the actual output) 

The difference between the predicted output and the actual output can be computed using the Kullback-Leibler (KL) divergence. 

We can then minimize the Kullback-Leibler (KL) divergence and simplify to get the formula for the negative log-likelihood criterion. 
