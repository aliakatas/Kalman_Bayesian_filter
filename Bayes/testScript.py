import numpy as np
from Bayes import Bayes

if __name__ == '__main__':

    # Define some generic values
    history = 10

    # Create some imaginary data
    obsNormal = np.random.normal(15.4, 2.3, size = 2000)
    modelNormal = obsNormal + np.random.normal(2.5, 0.5, size = len(obsNormal))

    # Define the plot range and size of the figure
    pltRange = range(1000, 1045)
    figSize = (11,9)

    # Start Testing
    # Create an instance of the object
    bs = Bayes(history)

    # Use the first #history of them for training
    obs_train = obsNormal[:history]
    model_train = modelNormal[:history]

    # The rest to be used dynamically
    obs_dyn = obsNormal[history:]
    model_dyn = modelNormal[history:]
    fcst = np.zeros_like(obs_dyn)

    # Perform an initial training of the model
    bs.trainMe(obs_train, model_train)
    


    