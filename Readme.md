# Neural Networks and Deep Learning Project Notes

   * [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
	  * [Review of the whole Project and Course](#review-of-the-whole-project-and-course)
      * [Python Basics with Numpy (optional assignment)](#python-basics-with-numpy-optional-assignment)
      * [Logistic Regression with a Neural Network mindset](#logistic-regression-with-a-neural-network-mindset)
      * [Planar data classification with one hidden layer](#planar-data-classification-with-one-hidden-layer)
      * [Building your Deep Neural Network: Step by Step](#building-your-deep-neural-network-step-by-step)
      * [Deep Neural Network for Image Classification: Application](#deep-neural-network-for-image-classification-application)
	  
This repository is the summaries of the project Neural Networks and Deep Learning on [DeepLearning.ai](https://deeplearning.ai) specialization courses.

## Neural Networks and Deep Learning

### Review of the whole Project and Course

> If you want to break into cutting-edge AI, this course will help you do so. Deep learning engineers are highly sought after, and mastering deep learning will give you numerous new career opportunities. Deep learning is also a new "superpower" that will let you build AI systems that just weren't possible a few years ago.
>
> In this course, you will learn the foundations of deep learning. When you finish this class, you will:
> - Understand the major technology trends driving Deep Learning
> - Be able to build, train and apply fully connected deep neural networks
> - Know how to implement efficient (vectorized) neural networks
> - Understand the key parameters in a neural network's architecture
>
> This course also teaches you how Deep Learning actually works, rather than presenting only a cursory or surface-level description. So after completing it, you will be able to apply deep learning to a your own applications. If you are looking for a job in AI, after this course you will also be able to answer basic interview questions.

### Python Basics with Numpy (optional assignment)

Welcome to your first assignment. This exercise gives you a brief introduction to Python. Even if you've used Python before, this will help familiarize you with functions we'll need.  

**Instructions:**
- You will be using Python 3.
- Avoid using for-loops and while-loops, unless you are explicitly told to do so.
- Do not modify the (# GRADED FUNCTION [function name]) comment in some cells. Your work would not be graded if you change this. Each cell containing that comment should only contain one function.
- After coding your function, run the cell right below it to check if your result is correct.

**After this assignment you will:**
- Be able to use iPython Notebooks
- Be able to use numpy functions and numpy matrix/vector operations
- Understand the concept of "broadcasting"
- Be able to vectorize code

Let's get started!

### Logistic Regression with a Neural Network mindset

Welcome to your first (required) programming assignment! You will build a logistic regression classifier to recognize  cats. This assignment will step you through how to do this with a Neural Network mindset, and so will also hone your intuitions about deep learning.

**Instructions:**
- Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.

**You will learn to:**
- Build the general architecture of a learning algorithm, including:
    - Initializing parameters
    - Calculating the cost function and its gradient
    - Using an optimization algorithm (gradient descent) 
- Gather all three functions above into a main model function, in the right order.

### Planar data classification with one hidden layer

Welcome to your week 3 programming assignment. It's time to build your first neural network, which will have a hidden layer. You will see a big difference between this model and the one you implemented using logistic regression. 

**You will learn how to:**
- Implement a 2-class classification neural network with a single hidden layer
- Use units with a non-linear activation function, such as tanh 
- Compute the cross entropy loss 
- Implement forward and backward propagation

### Building your Deep Neural Network: Step by Step

Welcome to your week 4 assignment (part 1 of 2)! You have previously trained a 2-layer Neural Network (with a single hidden layer). This week, you will build a deep neural network, with as many layers as you want!

- In this notebook, you will implement all the functions required to build a deep neural network.
- In the next assignment, you will use these functions to build a deep neural network for image classification.

**After this assignment you will be able to:**
- Use non-linear units like ReLU to improve your model
- Build a deeper neural network (with more than 1 hidden layer)
- Implement an easy-to-use neural network class

**Notation**:
- Superscript <sup>[l]</sup> denotes a quantity associated with the `l.th` layer.
    - Example: a<sup>[L]</sup> is the `L.th` layer activation. W<sup>[L]</sup> and b<sup>[L]</sup> are the `L.th` (last) layer parameters.
- Superscript <sup>(i)</sup> denotes a quantity associated with the `i.th` example.
    - Example: x<sup>(i)</sup> is the `i.th` training example.
- Lowerscript <sub>i</sub> denotes the `i.th` entry of a vector.
    - Example: a<sup>[l]</sup><sub>i</sub> denotes the `i.th` entry of the `l.th` layer's activations).

Let's get started!

### Deep Neural Network for Image Classification: Application

When you finish this, you will have finished the last programming assignment of Week 4, and also the last programming assignment of this course! 

You will use use the functions you'd implemented in the previous assignment to build a deep network, and apply it to cat vs non-cat classification. Hopefully, you will see an improvement in accuracy relative to your previous logistic regression implementation.  

**After this assignment you will be able to:**
- Build and apply a deep neural network to supervised learning. 

Let's get started!
