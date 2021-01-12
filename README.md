# Deep Neural Networks with PyTorch
<!-- GFM-TOC -->
* [1. Syllabus](#1-Syllabus)
    * [Chapter 1 Tensor and Datasets](#Chapter-1-Tensor-and-Datasets)
    * [Chapter 2 Linear Regression](#Chapter-2-Linear-Regression)
    * [Chapter 3 Linear Regression PyTorch Way](#Chapter-3-Linear-Regression-PyTorch-Way)
    * [Chapter 4 Multiple Input Output Linear Regression](#Chapter-4-Multiple-Input-Output-Linear-Regression)
    * [Chapter 5 Logistic Regression for Classification](#Chapter-5-Logistic-Regression-for-Classification)
    * [Chapter 6 Softmax Rergresstion](#Chapter-6-Softmax-Rergresstion)
    * [Chapter 7 Shallow Neural Networks](Chapter-7-Shallow-Neural-Networks)
    * [Chapter 8 Deep Networks](#Chapter-8-Deep-Networks)
    * [Chapter 9 Convolutional Neural Network](#Chapter-9-Convolutional-Neural-Network)
<!-- GFM-TOC -->
- [2. Improving Deep Neural Networks](#2-Improving-Deep-Neural-Networks)
  - [2.1 Practical Aspects of Deep Learning](#21-Practical-Aspects-of-Deep-Learning)
    - [2.1.1 Setting up your ML Application](#211-Setting-up-your-ML-Application)
    - [2.1.2 Regularizing your neural network](#212-Regularizing-your-neural-network)
    - [2.1.3 Setting up your optimization problem](#213-Setting-up-your-optimization-problem)
  - [2.2 Optimization Algorithms](#22-Optimization-Algorithms)
    - [2.2.1 Mini-Batch Gradient Descent](#221-Mini---Batch-Gradient-Descent)
    - [2.2.2 Exponentially Weighted Averages](#222-Exponentially-Weighted-Averages)
    - [2.2.3 Gradient Descent with Momemtum](#223-Gradient-Descent-with-Momentum)
    - [2.2.4 Root Mean Square Prop](#224-Root-Mean-Square-Prop)
    - [2.2.5 Adam Optimization Algorithm](#225-Adam-Optimization-Algorithm)
    - [2.2.6 Learning Rate Decay](#226-Learning-Rate-Decay) 
    - [2.2.7 The Problem of Local Optima](#227-The-Problem-of-Local-Optima)
  - [2.3 Tuning Process](#23-Tuning-Process)
    - [2.3.1 Hyperparameter Tuning](#231-Hyperparameter-Tuning)
    - [2.3.2 Batch Normalization](#232-Batch-Normalization)
    - [2.3.3 Softmax Regression](#233-Softmax-Regression)

This is an online course offered by Coursera. This course introduces how to develop deep learning models using Pytorch. 
Starting with the Pytorch's tensors, each section covers different models such as Linear Regression, and logistic/softmax regression.
Followed by Feedforward deep neural networks, the role of different activation functions, 
normalization and dropout layers. Then Convolutional Neural Networks and Transfer learning are also covered.

##  1. Syllabus
### Chapter 1 Tensor and Datasets
[1\. Tensor](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter01_01Tensor.py
)\
[2\. Dataset](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter01_02Dataset.py)\
[3\. Complex Dataset](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter01_03Complex_Dataset.py)

### Chapter 2 Linear Regression
[1\. Linear Regression with One Parameter](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter02_01LinearRegression_1P.py)\
[2\. Linear Regression - Prediction](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter02_02LinearRegression.py)\
[3\. Linear Regression with Two Parameters](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter02_03LinearRegression_2P.py)

### Chapter 3 Linear Regression PyTorch Way
[1\. Stochastic Gradient Descent](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter03_01StochasticGradientDescent.py)\
[2\. Linear Regression with SGD](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter03_02LR_SGD.py)\
[3\. Training and Validation Data with Pytorch](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter03_03Validation.py)

### Chapter 4 Multiple Input Output Linear Regression
[1\. Multiple Input Linear Regression ](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter04_01MultipleLR.py)\
[2\. Multiple Output Linear Regression](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter04_02MultipleOutputLR.py)\

### Chapter 5 Logistic Regression for Classification
[1\. Linear Classifier](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter05_01LinearClassifier.py)\
[2\. Logistic Regression](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter05_02LogisticRegression.py)

### Chapter 6 Softmax Rergresstion
[1\. Softmax in 1D](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter06_01Softmax1D.py)\
[2\. Softmax Classifier](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter06_02SoftmaxClassifier.py)

### Chapter 7 Shallow Neural Networks
[1\. Neural Network in 1D](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter07_01NN1D.py)\
[2\. Neural Network in 1D with Multiple Nodes](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter07_02NN1D_MultiNode.py)\
[3\. Neural Network with Multiple Dimensional Input](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter07_03NNMultiDim.py)\
[4\. Neural Network with Multiple Classes](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter07_04NNMultiClass.py)\
[5\. Activation Functions](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter07_05ActivationFunctions.py)

### Chapter 8 Deep Networks
[1\. Multiple Hidden Layer Deep Network](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_01MultiLayer.py)\
[2\. Multiple Hidden layer with ModuleList](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_02MultiLayer_ModuleList.py)\
[3\. Using Dropout for Classification](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_03Dropout.py)\
[4\. Using Dropout for Regression](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_04DropoutRegression.py)\
[5\. Initialization Weights](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_05InitializationWeights.py)\
[6\. Different Initialization Methods](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_06DifferentInitialization.py)\
[7\. Momentum](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_07Momentum.py)\
[8\. Neural Network with Momentum](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_08NNwithMomentum.py)\
[9\. Batch Normalization](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_09BatchNormalization.py)\
[10\. Batch Normalization and Dropout](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter08_10Batch%26Dropout.py)

### Chapter 9 Convolutional Neural Network
[1\. Convolution](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter09_01Convolution.py)\
[2\. Activation Function and Max Pooling](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter09_02ActFunc%26MaxPool.py)\
[3\. Multiple IO Channels](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter09_03MultiIN%26MultiOUT.py)\
[4\. Convolutional Neural Network](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter09_04CNN.py)\
[5\. CNN with MNIST](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter09_05CNNwithMNIST.py)\
[6\. CNN with Batch Normalization](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/Chapter09_06CNNwithBatchNormalization.py)


##  2. Improving Deep Neural Networks
### 2.1 Practical Aspects of Deep Learning
####  2.1.1 Setting up your ML Application
1. Seperate total data into training set, development set(for cross validation) and test set. Normally, they are seperated by 3/1/1 or, 7/3/0. When we have a huge size of data, we normally reduce the size of dev set and test set significantly.
2. Make sure the data in dev set and test set come from the same distribution. For an example, the pictures are in the training set are from experters, and the test pictures are from users using app and blurry.
3. Bias and variance. Underfitting => high bias, overfitting => high variance.\
When the error in the dev set is higher than that is in the training set, then the model is most likely overfitted with high variance. When the error in both dev and training set is high, then model is most likely underfitted with high bias. The model could also be with high bias (large error in training set )and high variance (large error in dev set).
4. Basic recipe for ML\
   4.1. High bias? (training set performance)  -->  Try bigger network/more layers/more neurons (may reduce bias without hurting variance), train longer or maybe other NN architecture\
   4.2. High variance? (dev set performance)  -->  More data (may reduce variance without hurting bias), regularization or maybe other NN architecture\
   4.3. Training a bigger network never hurts. The only drawback is the computational load.

####  2.1.2 Regularizing your neural network
1. L2 regularization is also named as weight decay. w_new = (1 - regularization term) * w_old - lr * dw.\
   It works just lik the ordinally gradient descent, where you update w by subtracting lr times the original gradient you got from backprop. But now you are also multiplying w by this thing, which is a little less than 1.
2. How does regularization prevent overfitting?\
   The extra regularization term penalizes the weight matrics from being too large.\
   Increasing the regularization term, we can reduce the value of the corresponding weight. When the regularization term is large, the weight is close to zero. Then that node will be zeroed out.
3. Dropout regularization (Inverted dropout)\
   3.1. Dropout is one of regularization methods to prevent overfitting.
   3.2. activation = activation / dropout prob. Make sure the expectation remains the same.\
4. Other regularization\
   4.1. Data augmentation, transform photos (flip, rotate or distortion)
   4.2. Early stopping. Stop training before dev set error gets larger.
   
####  2.1.3 Setting up your optimization problem
1. Normalizaing inputs will speed up training. Without normalizing inputs, we have to use small learning rate. If the features come from very different scales, then it's important to normalize features to help learning algorithm run faster.
2. Initialize weights properly can prevent vanishing or exploding gradients in a very deep network.


### 2.2 Optimization Algorithms
####  2.2.1 Mini-Batch Gradient Descent
1. One epoch denotes a single pass through training set.
2. Choosing mini-batch size:\
   2.1. If mini-batch size = training set size (m): batch gradient descent (low noise, relatively large step, take too long per iteration)\
   2.2. If mini-batch size is 1: stochastic gradietn descent (single step, can be extremely noisy, won't converge to the global minimum, just wonder round the region of minimum)\
   2.3. In pratice, mini-batch size is between 1 and m: fastest learning, make progress without processing entire training set\
   2.4. Small training set (<= 2000)- batch gradient descent\
   2.5. Mini-batch size is another hyperparameter, try different values and find out the one that makes the gradient descent optimization algorithm as efficient as possible.

####  2.2.2 Exponentially Weighted Averages
```
V_t = \beta * V_t-1 + (1 - \beta) * \theta_t
beta: weighting factor
theta_t: true value in the current time/step
V_0: assigned the initial weighted value, 0
V_t: weighted value in the current time/step
V_t-1: weighted value in the previous time/step
```
We can think V_t as approximately averaging over 1/(1-\beta). For example, if \beta = 0.9, we can think of this as averaging over the last 10 true values.\
For example:\
```
\beta = 0.9, V_0 = 0
V_100 = 0.1*\theta_100 + 0.1*0.9*\theta_99 + 0.1*0.9^2*\theta_98 + ... 
        + 0.1*0.9^n*\theta_(100-n) + ... + 0.1*0.9^99*\theta_1
The exponentially decaying function reduces from 0.1 to 0.1*0.9^99.
The way to compute V_100 is to take the element wise product between this 
exponentially decaying function and the true values and sum it up.
Note: 0.9^10 ~= 0.35 = 1/e, This means it takes about 10 steps for 
      the true value to decay to around 1/3 of the peak. 
      Therefore, when \beta = 0.9, this is as if we are computing an 
      exponentially weighted average that focused on just the last 10 true values.
```
1. One of the advantages of the exponentially weighted average, is that it takes very little memory. To calculate V, we only need to swap \theta once. On the contrary, to explicitly compute the average, we have to sum over the last n true values and divid by n. It requires more memery and is computational more expensive.\
2. Bias correction:\
   It turns out if we strictly implement the exponentially weighted average, the initial phase of V (the first a couple of values of V) will be much lower than the true values.\
   V_0 = 0, V_1 = 0.1*\theta_1, V_2 = 0.09*\theta_1 + 0.1*\theta_2 ...\
   To remove this bias, we can let:\
   V_t = V_t / (1 - \beta^t)
   
####  2.2.3 Gradient Descent with Momemtum
1. The basic idea is to compute the exponentially weighted average of the gradients, and use that gradient to update the weights instead. 
2. The conventional gradient descent method will the make the gradient slowly oscillate toward the minimum. This osillation slows down the gradient descent and prevents you from using a much larger learning rate.
3. Using gradient descent with momemtum reduces the oscillations and increase the convergence rate to the minimum.
```
On iteration t:
   Compute dW, db on the current mini-batch
   V_dW = \beta * V_dW + (1 - \beta) * dW 
   V_db = \beta * V_db + (1 - \beta) * db
   W := W - lr * V_dW
   b := b - lr * V_db
```

####  2.2.4 Root Mean Square Prop
To speed up the learning in the horizontal direction and slow down in the vertical direction.
```
Compute in the horizontal direction, normally small:
   S_dW = \beta * S_dW + (1 - \beta) * dW^2
Update W by dividing sqrt(S_dW), we increase the step size in the horizontal direction
   W := W - lr * dW/sqrt(S_dW)
Compute in the vertical direction, normally large
   S_db = \beta * S_db + (1 - \beta) * db^2
Update d by dividing sqrt(S_db), we reduce the step size (oscillations) in the vertical direction
   b := b - lr * db/sqrt(S_db)
To prevent to divid by 0, we normally add a very small term in the denominator, 
as sqrt(S_dW) + \epsilon.
```

####  2.2.5 Adam Optimization Algorithm
1. The basic idea of Adam (Adaptive Moment Estimation) optimization algorithm is taking momentum and RMSprop, and putting them together.
```
Initialize: V_dW = 0, S_dW = 0, V_db = 0, S_db = 0
On iteration t:
   Compute dW, db using current mini-batch
   Implement momentum (\beta1):
      V_dW = \beta1 * V_dW + (1 - \beta1) * dW
      V_db = \beta1 * V_db + (1 - \beta1) * db
   Implement RMSprop (\beta2):
      S_dW = \beta2 * S_dW + (1 - \beat2) * dW^2
      S_db = \beta2 * S_db + (1 - \beta2) * db^2
   Implement Adam algorithm:
      V^corrected_dW = V_dW / (1 - \beta1^t),  V^corrected_db = V_db / (1 - \beta1^t)
      S^corrected_dW = S_dW / (1 - \beta2^t),  S^corrected_db = S_db / (1 - \beta2^t)
   Update coefficients:
      W := W - lr * V^corrected_dW / (sqrt(S^corrected_dW) + \epsilon)
      b := b - lr * V^corrected_db / (sqrt(S^corrected_db) + \epsilon)
  
Hyperparameters choice:
   lr: needs to be tune
   \beta1: 0.9
   \beta2: 0.999
   \epsilon: 10^-8
```

####  2.2.6 Learning Rate Decay
1. The basic idea is to slowly reduce the learning rate as approaching toward the minimum.
```
lr = 1 / (1 + decay rate * epoch num) * lr0
For example:
lr0 = 0.2, decay rate = 1
epoch num  |  lr
    1         0.1
    2         0.067
    3         0.05
    4         0.04
```
2. Exponential day
```
1. lr = 0.95^epoch-num * lr0
2. lr = k / sqrt(epoch-num) * lr0
3. Manual decay
```

####  2.2.7 The Problem of Local Optima
1. Most points of zero gradient are not local optima, but saddle points. In a high dimentional space, the gradient at the local optima must be zero in each of the dimention. This chance of that happening is very small.
2. The problem of plateaus. A  plateau is a region where the derivative is close to zero for a long time. This is where algorithms like momentum or RmsProp or Adam can really help your learning algorithm.



### 2.3 Tuning Process
#### 2.3.1 Hyperparameter Tuning
1. Learning rate, momemtum (0.9 is a good choice), # of hidden units, mini-batch size, # of layers, learning rate decay and etc.
3. When the # of hyperparameters is small, we can use a grid. But random sampling is recommended.
3. Using multifidelity to randomly select hyperparameters. First, sample in a large range of values. Second, once find out the region in which the hyperparameters work better than other areas, we zoom in that region and keep sampling randomly.
4. Using an appropriate scale to pick hyperparameters. Instead of sampling uniformly random, we can sample in the log scale.\
    Example1, 0.0001, 0.001, 0.01, 0.1. Take the log of those values, we have 4, 3, 2, 1.\
    Example2, select hyperparameter for exponentially weighted averages. \beta = 0.9, ..., 0.999, we can take the log of (1 - \beta). The reason is 1/ (1 - \beta) gets very sensitive when \beta is close to 1. We need to sample more densely in the region of when \beta is close to 1.
5. If you have enough computers to train a lot of models in parallel, then try a lot of different hyperparameters together. On the other hand, tune the hyperparameters up and down liking babysitting a model. 

#### 2.3.2 Batch Normalization
1. Normalize the values before the activation layer.
```
Given some intermediate values in the neural network, z_1, ..., z_m
1. \mu = \sum (z_i) / m
   \sigma^2 = \sum (z_i - \mu)^2 / m
   z_norm = (z - \mu) / sqrt(\sigma^2 + \epsilon), mean zero and variance one

2. z_tilt = \gamma * z_norm + \beta, \gamma and \beta are learnable variables
   Using this function, we can normalize around other value with other variance.
```
2. Batch norm processes data one mini batch at a time.
3. Reduce the amount that the distribution of hidden unit values shifts around. In other words, it reduces the problem of the input values changing.
4. Each mini-batch is scaled by the mean/variance computed on just that mini-batch. This adds some noise to the values z within that mini-batch. Because by adding noise to the hidden units, it's forcing the downstream hidden units not to rely too much on any one hidden unit. So similar to dropout, it adds some noise to each hidden layer's activations. This has a slight regularization effect. Use a big size of mini-batch, will reduce the regularization effect.
5. Batch norm at test time. For different mini batch, we calculate the mean and variance. In addition,  we keep tracking the mean and variance. At test time, we estimate the mean and variance value using exponentially weighted average (across mini-batch). We then use these estimated values to compute the z_norm at test time.


#### 2.3.3 Softmax Regression
```
On the lay l:
   z^[l] = w^[l] * a^[l-1] + b^[l]
Activation function:
   t^[l] = exp(z^[l])
   a^[l] = t^[l]_i / \sum(t^[l]_i),  i = 1, ..., # of hidden nodes
```
1. Softmax regression generalizes logistic regression to C classes. If C = 2, softmax regression reduces to logistic regression.
2. Loss function:
   L(yhat, y) = - \sum y_j * log(yhat_j), j = 1, ..., # of classes
3. Cost function:
   J = \sum L(yhat_i, y_i) / m, i = 1, ..., m (# of samples)



















