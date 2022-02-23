<!-- GFM-TOC -->
* [1. Convolution](#1-Convolution)
* [2. Perceptron](#2-Perceptron)
* [3. Loss function](#3-Loss-function)


# 1. Convolution
## What is convolution?
Convolution can interpret a process that consists of two functions.\
Specifically, the convolution is normally expressed as:\
![Figure 1](https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSeBqgdnEmDF2l2BV7fVkZHmK4MHdIHRqZq7jCf2Hu3JyErSnfm)

One typical example is shown below.
![image1](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/images/Convolution.png)

In the image illustrated above, function, f(t), represents the food that we take at time \tau, and the function, g(t), interprets the progress of digestion of each food that we take. For example, suppose we eat a bowl of rice at 12 o'clock, then the amount of rice left in our stomach at 14 o'clock is computed by f(12)g(14-12).\
As a result, the convolution between f(t) and g(t) can be interpreted as calculating the total amount of food that are eaten before time t will be left in the stomach at time t.\

## What can convolution do in the system?
For a system, if the input is not stable (i.e., f(t) in the above example), but the output is stable (i.e., g(t) in the above example), then we can employ convolution to calculate the amount of information saved/left in the system.

## What is convolution in the CNN or in the image processing?
For CNN, we may think of the input image as the unstable input, f(t), and the kernel as the stable output, g(t).\
We use a kernel to sweep through the whole image. For each round of calculation, we compute the summation of the product of the corresponding cell. The more similar between the kernel and the particular part of image, the higher convolution result we can obtain. In other words, the convolution in the CNN looks for the pattern that is similar to the kernel.

# 2. Perceptron
Perceptron is a method to seperate a dataset into two individual sets. It works for AND, OR, and NOT, but not for XOR, as shown in the figure below.\
![image1](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/images/Perceptron.png)

To deal with this issue, we can employ multiple perceptrons as below,\
![image2](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/images/Multiple_perceptron.png)

or increase the dimension of the dataset as shown below\
![image3](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/images/Increase_dimension.png)

# 3. Loss function
The minimum mean squared error is not a very good option for gradient descent method. Compared to minimum squared error, the maximum likelihood is a proper option.\
![image4](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/images/Maximum_likelihood.png)

As shown in the figure above, the basic idea of the maximum likelihood is to find the best distribution over the actual outputs according to the actual outputs. In this example, the probability 0.7 for the head and 0.3 for the tail should be the best distribution to describe the outputs. In other words, we try to calculate the probability of the outputs conditioned on the prior distribution.\
![image5](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/images/Maximum_likelihood2.png)





