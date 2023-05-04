import numpy as np
import matplotlib.pyplot as plt
def visualize(i, data):
   
    image = data[0][i]

    # Reshape the image data into a 28x28 array
    image = image.reshape(28, 28)

    # Display the image using Matplotlib
    plt.imshow(image, cmap='gray')
    plt.title(np.argmax(data[1][i]))
    plt.show()