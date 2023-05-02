import network0
import numpy as np
import logging

# Create a logger object
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler and set the logging level
file_handler = logging.FileHandler('shape_logging.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger object
logger.addHandler(file_handler)


# function that checks if weights matrix has right dimensions
def check_weights(net: network0.network, test: np.array):
    for i in range(len(net.layers)-1):
        if not (test[i].shape == (net.layers[i+1], net.layers[i])):
            logger.debug(f"weights check: test shape at index {i}: {test[i].shape} | weights shape at index {i}: {net.weights[i].shape}")
            logger.warn(f"test: {test} | weights: {net.weights}")
            assert(0)

def check_biases(net: network0.network, test: np.array):
    for i in range(1, len(net.layers)):
        if not (test[i-1].shape == (net.layers[i])):
            logger.debug(f"biases check: test shape at index {i-1}: {test[i-1].shape} | biases shape at index {i-1}: {net.biases[i-1].shape}")
            logger.warn(f"test: {test} | biases: {net.biases}")
            assert(0)