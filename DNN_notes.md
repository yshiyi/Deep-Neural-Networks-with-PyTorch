<!-- GFM-TOC -->
* [1. Convolution](#1-Convolution)


# 1. Convolution
## What is convolution?
Convolution can interpret a process that consists of two functions.\
Specifically, the convolution is normally expressed as:\
![Figure 1](https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcSeBqgdnEmDF2l2BV7fVkZHmK4MHdIHRqZq7jCf2Hu3JyErSnfm)

One typical example is shown below.
![image](https://github.com/yshiyi/Deep-Neural-Networks-with-PyTorch/blob/main/images/Convolution.png)

In the image illustrated above, function, f(t), represents the food that we take at time \tau, and the function, g(t), interprets the progress of digestion of each food that we take. For example, suppose we eat a bowl of rice at 12 o'clock, then the amount of rice left in our stomach at 14 o'clock is computed by f(12)g(14-12).\
As a result, the convolution between f(t) and g(t) can be interpreted as calculating the total amount of food that are eaten before time t will be left in the stomach at time t.\

## What can convolution do in the system?
For a system, if the input is not stable (like function f(t) in ), but the output is stable, then we can employ convolution to calculate the amount of information saved/left in the system.
