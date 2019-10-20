import numpy as np
from Bayes import Bayes

if __name__ == '__main__':

    # Define some generic values
    history = 7

    # Create some imaginary data
    obsNormal = np.random.normal(15.4, 2.3, size = 2000)
    modelNormal = obsNormal + np.random.normal(2.5, 0.5, size = len(obsNormal))

    