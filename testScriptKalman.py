"""
Testing Kalman filter for randomly generated data.
"""

import numpy as np
from Kalman import Kalman
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Define some generic values
    history = 7
    order = 2
    totNum = 2000
    obsMean = 15.4
    obsVar = 2.3
    modBias = 2.5
    modVar = 0.5

    pltStart = 1000
    pltNum = 45
    figSize = (11,9)

    # Create some imaginary data
    obs = np.random.normal(obsMean, obsVar, size = totNum)
    model = obs + np.random.normal(modBias, modVar, size = len(obs))

    # Define the plot range and size of the figure
    pltRange = range(pltStart, pltStart + pltNum)
    
    # Start Testing
    # Create an instance of the filter
    kf = Kalman(history, order)
    
    # Use the first #history of them for training
    obs_train = obs[:history]
    model_train = model[:history]

    # The rest to be used dynamically
    obs_dyn = obs[history:]
    model_dyn = model[history:]
    fcst = np.zeros_like(obs_dyn)

    # Perform an initial training of the model
    kf.trainMe(obs_train, model_train)

    for ij in range(len(obs_dyn)):
        # Provide a correction to the forecast
        fcst[ij] = kf.adjustForecast(model_dyn[ij])

        # Update filter
        kf.trainMe([obs_dyn[ij]], [model_dyn[ij]])

    fig = plt.figure(figsize=figSize)
    plt.plot(obs_dyn[pltRange], label='obs')
    plt.plot(model_dyn[pltRange], '--', label='model')
    plt.plot(fcst[pltRange], '*', label='kalman')
    plt.legend()
    plt.show()


    

    