## Week 1 Quiz - Introduction to deep learning

1. What does the analogy “AI is the new electricity” refer to?

    - [ ] AI is powering personal devices in our homes and offices, similar to electricity.
    - [ ] Through the “smart grid”, AI is delivering a new wave of electricity.
    - [ ] AI runs on computers and is thus powered by electricity, but it is letting computers do things not possible before.
    - [x] Similar to electricity starting about 100 years ago, AI is transforming multiple industries.
    
    Note: Andrew illustrated the same idea in the lecture.

2. Which of these are reasons for Deep Learning recently taking off? (Check the two options that apply.)

    - [x] We have access to a lot more computational power.
    - [ ] Neural Networks are a brand new field.
    - [x] We have access to a lot more data.
    - [x] Deep learning has resulted in significant improvements in important applications such as online advertising, speech recognition, and image recognition.
    
3. Recall this diagram of iterating over different ML ideas. Which of the statements below are true? (Check all that apply.)

    - [x] Being able to try out ideas quickly allows deep learning engineers to iterate more quickly.
    - [x] Faster computation can help speed up how long a team takes to iterate to a good idea. 
    - [ ] It is faster to train on a big dataset than a small dataset.
    - [x] Recent progress in deep learning algorithms has allowed us to train good models faster (even without changing the CPU/GPU hardware).

    Note: A bigger dataset generally requires more time to train on a same model.

4. When an experienced deep learning engineer works on a new problem, they can usually use insight from previous problems to train a good model on the first try, without needing to iterate multiple times through different models. True/False?

    - [ ] True
    - [x] False
    
    Note: Maybe some experience may help, but nobody can always find the best model or hyperparameters without iterations. 

5. Which one of these plots represents a ReLU activation function?

    - Check [relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
    
6. Images for cat recognition is an example of “structured” data, because it is represented as a structured array in a computer. True/False?
    
    - [ ] True
    - [x] False
    
7. A demographic dataset with statistics on different cities' population, GDP per capita, economic growth is an example of “unstructured” data because it contains data coming from different sources. True/False?
    
    - [ ] True
    - [x] False
    
8. Why is an RNN (Recurrent Neural Network) used for machine translation, say translating English to French? (Check all that apply.)

    - [x] It can be trained as a supervised learning problem.
    - [ ] It is strictly more powerful than a Convolutional Neural Network (CNN).
    - [x] It is applicable when the input/output is a sequence (e.g., a sequence of words).
    - [ ] RNNs represent the recurrent process of Idea->Code->Experiment->Idea->....
    
9. In this diagram which we hand-drew in lecture, what do the horizontal axis (x-axis) and vertical axis (y-axis) represent?

    - x-axis is the amount of data
    - y-axis (vertical axis) is the performance of the algorithm.

10. Assuming the trends described in the previous question's figure are accurate (and hoping you got the axis labels right), which of the following are true? (Check all that apply.)

    - [x] Increasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.
    - [x] Increasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.
    - [ ] Decreasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.
    - [ ] Decreasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.
	
## Week 2 Quiz - Neural Network Basics

1. What does a neuron compute?

    - [ ] A neuron computes an activation function followed by a linear function (z = Wx + b)

    - [x] A neuron computes a linear function (z = Wx + b) followed by an activation function

    - [ ] A neuron computes a function g that scales the input x linearly (Wx + b)

    - [ ] A neuron computes the mean of all features before applying the output to an activation function

    Note: we generally say that the output of a neuron is a = g(Wx + b) where g is the activation function (sigmoid, tanh, ReLU, ...).
    
2. Which of these is the "Logistic Loss"?

    - Check [here](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression).
    
    Note: this is the logistic loss you've seen in lecture!
    
3. Suppose img is a (32,32,3) array, representing a 32x32 image with 3 color channels red, green and blue. How do you reshape this into a column vector?

    - `x = img.reshape((32 * 32 * 3, 1))`
    
4. Consider the two following random arrays "a" and "b":

    ```
    a = np.random.randn(2, 3) # a.shape = (2, 3)
    b = np.random.randn(2, 1) # b.shape = (2, 1)
    c = a + b
    ```
    
    What will be the shape of "c"?
    
    b (column vector) is copied 3 times so that it can be summed to each column of a. Therefore, `c.shape = (2, 3)`.
    
    
5. Consider the two following random arrays "a" and "b":

    ```
    a = np.random.randn(4, 3) # a.shape = (4, 3)
    b = np.random.randn(3, 2) # b.shape = (3, 2)
    c = a * b
    ```
    
    What will be the shape of "c"?
    
     "*" operator indicates element-wise multiplication. Element-wise multiplication requires same dimension between two matrices. It's going to be an error.

6. Suppose you have n_x input features per example. Recall that X=[x^(1), x^(2)...x^(m)]. What is the dimension of X?

    `(n_x, m)`

    
7. Recall that `np.dot(a,b)` performs a matrix multiplication on a and b, whereas `a*b` performs an element-wise multiplication.

    Consider the two following random arrays "a" and "b":

    ```
    a = np.random.randn(12288, 150) # a.shape = (12288, 150)
    b = np.random.randn(150, 45) # b.shape = (150, 45)
    c = np.dot(a, b)
    ```
    
    What is the shape of c?
    
    `c.shape = (12288, 45)`, this is a simple matrix multiplication example.
    
8. Consider the following code snippet:

    ```
    # a.shape = (3,4)
    # b.shape = (4,1)
    for i in range(3):
      for j in range(4):
        c[i][j] = a[i][j] + b[j]
    ```
    
    How do you vectorize this?

    `c = a + b.T`

9. Consider the following code:

    ```
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 1)
    c = a * b
    ```
    
    What will be c?
    
    This will invoke broadcasting, so b is copied three times to become (3,3), and ∗ is an element-wise product so `c.shape = (3, 3)`.
    
10. Consider the following computation graph.

    ```
    J = u + v - w
      = a * b + a * c - (b + c)
      = a * (b + c) - (b + c)
      = (a - 1) * (b + c)
    ```
      
    Answer: `(a - 1) * (b + c)`
	
## Week 3 Quiz -  Shallow Neural Networks

1. Which of the following are true? (Check all that apply.) **Notice that I only list correct options.**

    - X is a matrix in which each column is one training example.
    - a^[2]_4 is the activation output by the 4th neuron of the 2nd layer
    - a^\[2\](12) denotes the activation vector of the 2nd layer for the 12th training example.
    - a^[2] denotes the activation vector of the 2nd layer.

2. The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False?

    - [x] True
    - [ ] False
    
    Note: You can check [this post](https://stats.stackexchange.com/a/101563/169377) and (this paper)[http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf].
    
    > As seen in lecture the output of the tanh is between -1 and 1, it thus centers the data which makes the learning simpler for the next layer.
    
3. Which of these is a correct vectorized implementation of forward propagation for layer l, where 1≤l≤L?

    - Z^[l]=W^[l]A^[l−1]+b^[l]
    - A^[l]=g^\[l](Z^[l])

4. You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer?

    - [ ] ReLU
    - [ ] Leaky ReLU
    - [x] sigmoid
    - [ ] tanh
    
    Note: The output value from a sigmoid function can be easily understood as a probability.
    
    > Sigmoid outputs a value between 0 and 1 which makes it a very good choice for binary classification. You can classify as 0 if the output is less than 0.5 and classify as 1 if the output is more than 0.5. It can be done with tanh as well but it is less convenient as the output is between -1 and 1.
    
5. Consider the following code:

    ```
    A = np.random.randn(4,3)
    B = np.sum(A, axis = 1, keepdims = True)
    ```
    
    What will be B.shape?
    
    `B.shape = (4, 1)`
    
    >  we use (keepdims = True) to make sure that A.shape is (4,1) and not (4, ). It makes our code more rigorous.

6. Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements are True? (Check all that apply)

    - [x] Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons.
    - [ ] Each neuron in the first hidden layer will perform the same computation in the first iteration. But after one iteration of gradient descent they will learn to compute different things because we have “broken symmetry”.
    - [ ] Each neuron in the first hidden layer will compute the same thing, but neurons in different layers will compute different things, thus we have accomplished “symmetry breaking” as described in lecture.
    - [ ] The first hidden layer’s neurons will perform different computations from each other even in the first iteration; their parameters will thus keep evolving in their own way.
    
7. Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?

    - [ ] True
    - [x] False
    
    >  Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector.

8. You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen?

    - [ ] It doesn’t matter. So long as you initialize the weights randomly gradient descent is not affected by whether the weights are large or small.

    - [ ] This will cause the inputs of the tanh to also be very large, thus causing gradients to also become large. You therefore have to set α to be very small to prevent divergence; this will slow down learning.

    - [ ] This will cause the inputs of the tanh to also be very large, causing the units to be “highly activated” and thus speed up learning compared to if the weights had to start from small values.

    - [x] This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow.


    > tanh becomes flat for large values, this leads its gradient to be close to zero. This slows down the optimization algorithm.
    
9. Consider the following 1 hidden layer neural network:

    - b[1] will have shape (4, 1)
    - W[1] will have shape (4, 2)
    - W[2] will have shape (1, 4)
    - b[2] will have shape (1, 1)
    
    Note: Check [here](https://user-images.githubusercontent.com/14886380/29200515-7fdd1548-7e88-11e7-9d05-0878fe96bcfa.png) for general formulas to do this.
    
10. In the same network as the previous question, what are the dimensions of Z^[1] and A^[1]?

    - Z[1] and A[1] are (4,m)
    
    Note: Check [here](https://user-images.githubusercontent.com/14886380/29200515-7fdd1548-7e88-11e7-9d05-0878fe96bcfa.png) for general formulas to do this.
	
## Week 4 Quiz - Key concepts on Deep Neural Networks

1. What is the "cache" used for in our implementation of forward propagation and backward propagation?

    - [ ] It is used to cache the intermediate values of the cost function during training.
    - [x] We use it to pass variables computed during forward propagation to the corresponding backward propagation step. It contains useful values for backward propagation to compute derivatives.
    - [ ] It is used to keep track of the hyperparameters that we are searching over, to speed up computation.
    - [ ] We use it to pass variables computed during backward propagation to the corresponding forward propagation step. It contains useful values for forward propagation to compute activations.

    > the "cache" records values from the forward propagation units and sends it to the backward propagation units because it is needed to compute the chain rule derivatives.

2. Among the following, which ones are "hyperparameters"? (Check all that apply.) **I only list correct options.**

    - size of the hidden layers n^[l]
    - learning rate α
    - number of iterations
    - number of layers L in the neural network

    Note: You can check [this Quora post](https://www.quora.com/What-are-hyperparameters-in-machine-learning) or [this blog post](http://colinraffel.com/wiki/neural_network_hyperparameters).
    
3. Which of the following statements is true?

    - [x] The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.
Correct 
    - [ ] The earlier layers of a neural network are typically computing more complex features of the input than the deeper layers.
    
    Note: You can check the lecture videos. I think Andrew used a CNN example to explain this.

4. Vectorization allows you to compute forward propagation in an L-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False?

    - [ ] True
    - [x] False
    
    Note: We cannot avoid the for-loop iteration over the computations among layers.
    
5. Assume we store the values for n^[l] in an array called layers, as follows: layer_dims = [n_x, 4,3,2,1]. So layer 1 has four hidden units, layer 2 has 3 hidden units and so on. Which of the following for-loops will allow you to initialize the parameters for the model?

    ```
    for(i in range(1, len(layer_dims))):
        parameter[‘W’ + str(i)] = np.random.randn(layers[i], layers[i - 1])) * 0.01
        parameter[‘b’ + str(i)] = np.random.randn(layers[i], 1) * 0.01
    ```

6. Consider the following neural network.

    - The number of layers L is 4. The number of hidden layers is 3.
    
    Note: The input layer (L^[0]) does not count.
    
    > As seen in lecture, the number of layers is counted as the number of hidden layers + 1. The input and output layers are not counted as hidden layers.

7. During forward propagation, in the forward function for a layer l you need to know what is the activation function in a layer (Sigmoid, tanh, ReLU, etc.). During backpropagation, the corresponding backward function also needs to know what is the activation function for layer l, since the gradient depends on it. True/False?

    - [x] True
    - [ ] False
    
    > During backpropagation you need to know which activation was used in the forward propagation to be able to compute the correct derivative.
    
8. There are certain functions with the following properties:

    (i) To compute the function using a shallow network circuit, you will need a large network (where we measure size by the number of logic gates in the network), but (ii) To compute it using a deep network circuit, you need only an exponentially smaller network. True/False?
    
    - [x] True
    - [ ] False
    
    Note: See lectures, exactly same idea was explained.
    
9. Consider the following 2 hidden layer neural network:

    Which of the following statements are True? (Check all that apply).

    - W^[1] will have shape (4, 4)
    - b^[1] will have shape (4, 1)
    - W^[2] will have shape (3, 4)
    - b^[2] will have shape (3, 1)
    - b^[3] will have shape (1, 1)
    - W^[3] will have shape (1, 3)
    
    Note: See [this image](https://user-images.githubusercontent.com/14886380/29200515-7fdd1548-7e88-11e7-9d05-0878fe96bcfa.png) for general formulas.
    
    
10. Whereas the previous question used a specific network, in the general case what is the dimension of W^[l], the weight matrix associated with layer l?

    - W^[l] has shape (n^[l],n^[l−1])
    
    Note: See [this image](https://user-images.githubusercontent.com/14886380/29200515-7fdd1548-7e88-11e7-9d05-0878fe96bcfa.png) for general formulas.	