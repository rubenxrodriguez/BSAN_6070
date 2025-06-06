# 1. Neural Network Fundamentals

## 1.1 Network Architecture
- A Neuron is a weighted linear combination of observations followed by a non-linear tranformation by an "activation function"
  * Y = predicted = softmax (f'(f(X,W1),W2))
  * f and f' are typically sigmoid functions that map values to 0-1. Then we take a softmax of that. 
- Layer Types : 
  * Input Layer:
    - Receives raw features (e.g., word embeddings, pixel values)
    - Dimension matches feature space
  * Hidden Layers :
    - Extract hierarchical patterns (shallow -> deep features)
    - Typical architectures : Dense, Convolutional, Recurrent
    - Final prediction (e.g., softmax for classification, linear for regression) 
    - Size matches task (e.g., 1 for binary, N for multi-class)
  * Information Flow :
    - Feedforward: Data moves input -> hidden -> output
    - Transformations: Matrix multiplications + activation functions
## 1.3 Loss Function
  * A gradient of the loss function with respect to a trainable parameter will point in the direction of greatest decrease of the loss function
  * Stochastic gradient descent updates weights using the gradient from a single random example per step
        * decrease the learning rate over time to stabilize convergence
  * ADAM (Adaptive Moment Estimation)
        * tracks exponentially weighted average of past gradients
        * adaptive learning rate - based on average of past squared gradients
## 1.2 Activation Functions
  * Role:
      - Introduce non-linearity (enables complex function approximation)
      - Control neuron output range (e.g., [0,1] for sigmoid)
  * Common Types:
      - ReLu: Avoids vanishing gradients in deep networks
      - Sigmoid: For probabilistic outputs (binary classification)
      - Softmax: Multi-class probability distributions
## 1.3 Backpropagation
  * Process:
      - Forward pass: Compute predictions and loss
      - Backward pass: Calculate gradients via chain rule
          * Propogates error from output -> input layers
      - Weight updates: Adjust parameters via optimizer (e.g., SGD, Adam)
  * Importance:
      - Enables end-to-end learning in deep networks
      - Efficient gradient computation for millions of parameters
      - resuses computations from the forward pass, making it orders of magnitude faster
## 1.4 Types of ANN
  * Feed Forward Neural Network (FF-ANN)
      - All nodes are fully connected
      - At least ONE hidden layer
      - Information only moves forward without any back loop
      - feedforward only refers to inference direction, not training (meaning there is backprop)
  * Deep FF-ANN (DFF-ANN)
      - An FF-ANN with more than one hidden layer
      - Provides excellend granularity in decision but at the cost of exponential growth in training time
  * Radial Basis Function Network (RBF-ANN)
      - Multi-layered FF-ANN
      - Activation Function is Radial Basis Function [ not restricted to 0-1 (like Sigmoid) ]
      - useful in predicting continuous outcomes like sales,income,house price (like regression)
  * Recurrent Neural Network (RNN)
      - variation of FF-ANN where nodes can receive its own output as an input with delay
      - Widely used in applications where "context" is important
          * text analytics and natural language processing
  * Convolutional Neural Network (CNN)
      - Image recognition and processing.
      - Uses convolutional operator
      - Other layeres such as ReLU, Pooling & Classfication
  * Long/Short Term Memory Network (LSTM)
      - Incoroporation of "memory cell" concept
      - Can process "longitudinal data" or time interval data
      - Widely used in speech recognition, voice-to-text
## 1.5 Comparison with Traditional ML
  * Advantages:
      - Automatic feature engineering (vs manual feature selection)
      - Handles unstructured data (text, images) effectively
      - Can be used for both regression/classfication
      - Flexibility and customizability with layers
      - ANNs are very good at handling non-linear data with a large number of features. This is especially good for image processing, NLP, as well as predictive business applications
      - ANN is memory resident and provides some amount of fault tolerance
      - Once trained, predictions are fast
  * Disadvantages:
      - Requires large amount of labelled data. This exposes ANNs to overfitting risks.
      - Black-box nature (less interpretable than decision trees)


# 2. Recurrent Neural Networks (RNNs)
## 2.1 Core Concepts
  * One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task
    - They are networks with loops in them, allowing information to persist
  * Can be thought of as multiple copies of the same network, each passing a message to a successor
  * Sequential Processing:
    - Maintains hidden state vector (memory) across time steps 
    - Processes variable-length sequences (e.g. sentences, time-series)
  * Vanishing Gradients:
    - Problem: Gradient shrink exponentially in deep/time steps
    - Impact: Fails to learn long-range dependencies (>10 steps) ; unable to connect the information ; LSTMs solve this problem
## 2.2 Business Applications
  * Speech Recognition, Language Modeling, Translation, Image Captioning
  * Time-Series:
      - Demand for forecasting (weights recent observations more)
      - Anomaly detection in sensor data
  *  Sentiment analysis (context-aware classification)


# 3. LSTMs & Advanced RNNs
## 3.1 Architecture
  * Explicitly designed to avoid the long-term dependency problem
  * All RNNs have the form of a chain of repeating modules of neural network
      - in standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer
      - LSTMs have four neural network layers interacting in a very special way
  * LSTMs have a new mechanism **Memory Cells**
      - more analogous to a "pipe". What flows through the pipe is what you need to remember and use
      - you can modify this flow by adding flow("remember") or flow("forget") to reduce the flow
  * LSTMs make decisions using (a) Current Inputs (b) Previous Outputs (c) Previous Memory
  * It generates (a) New Outputs (decision) (b) Modification of Memory
  
## 3.2 Advanced Topics
   - LSTM has the ability to remove or add information to the cell state - careufllly regulated by structures called "gates"
      - they are composed out a sigmoid layer and a pointwise multiplication operation.
      - gates are a way to optionally let information through
   - The first step is to decide what information we're going to throw away from the cell state - "forget gate layer"
      - For a language model it might be trying to predict the next word based on all the previous ones. When we see a new subject, we want to forget the gender of the old subject ("John plays golf with Linda. ___ was a beginner")
   - Next step is to decide what new infomration is gonig to be stored in the cell state. This has two parts.
     - First, a sigmoid called "input gate layer" decides what values we'll update (this is for the main loop)
     - Next, a tanh layer creates a vector of new candidates ~C_t that could be added to the memory state.
   - Now, it's time to update the old cell state C_{t-1} to C_t
     - C_t = f_t (forget value from forget gate layer) * C_{t-1} + i_t (sigmoid value from input gate layer) * ~C_t.
   - Finally, we need to decide what to output
## 3.3 Advantages
   - Abilitity to remember or forget over extended time intervals
   - Robustness to noise
   - Adaptable: can handle sequences of variable length and can take multiple time steps as input
   - Suitable for multi-step forecasting: LSTM can be trained to predict multiple outputs at once, making them well-suited for multi-step forecasting
   - Ability to learn complex patterns: capable of learning complex non-linear patterns 
## 3.4 Example Application
   - Stock Prediction: Learns to retain key trends (50-day MA) while ignoring noise
## 3.5 Limitations
  * Computational Cost:
      - 3-4x more parameters than simple RNNs
      - Slow training for very long sequences (>1000 steps)
