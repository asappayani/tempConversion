import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plot
import logging

""" 
Supervised learning essentially takes in inputs and outputs and finds an algorithm. 
For this project, I'm training this model to create an algorithm that'll convert celsius to fahrenheit 
"""

logger = tf.get_logger()
logger.setLevel(logging.ERROR) # This will make it only display errors

# TRAINING DATA, this will be the data I use to train model
celsius = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)


# CREATING LAYERS

"""
LAYERS, b/c this is a simple problem, we'll only need one layer with one neuron
input_shape = (1,) means that there is only one input to the layer. 
units = 1 means the number of neurons in the layer 
"""

layer1 = tf.keras.layers.Dense(units=1, input_shape=(1,))

# CREATING THE MODEL

"""
After defining layers, they need to be assembled as a model using the sequential function which takes a list of layers
Typically, we'll see the layers creaeted inside the model instead of beforehand
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])
"""

model = tf.keras.Sequential([layer1])

# COMPILING THE MODEL, just adding a loss and optimizer function

"""
A loss function measures how far off predictions are from the desired outcome
There are multiple types of loss functions (should probably research them when I can), and I'll use the mean squared error for this model
The optimizer is Adam which is built into tensorflow (probably research how it works too), the learning rate which is the 0.1 is the step size (wtv that means)
taken when adjusting values in the model, if it's too small then it'll run too many iterations to train model, too big and it'll be very inaccurate. Range is usually between .001 and 0.1
"""

model.compile(loss="mean_squared_error",
              optimizer=tf.keras.optimizers.Adam(0.1))

# TRAINING THE MODEL, using the fit function

"""
The epochs argument is how many times the cycle (calculate -> compare -> adjust) is ran, verbose controls how many output the method produces
"""

history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

# DISPLAY TRAINING STATISTICS

"""
# This graph will show how the loss of the model decreases the more it trains after each epoch
"""

plot.xlabel("Epoch")
plot.ylabel("Loss Magnitude")
plot.plot(history.history["loss"])

# MODEL PREDICTIONS
print(model.predict(np.array([100.0]))) # Now that the model is trained, the predict function can be used to make predictions


# THE MODEL'S WEIGHTS (research what these are b/c i'm confused)
print(f'These are the layer weights: {layer1.get_weights()}')