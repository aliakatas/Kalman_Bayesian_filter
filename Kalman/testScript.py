import numpy as np
from Kalman import Kalman
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Define some generic values
    history = 7
    order = 2

    # Create some imaginary data
    obs = np.random.normal(15.4, 2.3, size = 2000)
    model = obs + np.random.normal(2.5, 0.5, size = len(obs))

    # Define the plot range and size of the figure
    pltRange = range(1000, 1045)
    figSize = (11,9)

    # Start Testing
    # Create an instance of the filter
    kf = Kalman(history, order)
    
    # Use the first 7 of them for training
    obs_train = obs[:history]
    model_train = model[:history]

    # The rest to be used dynamically
    obs_dyn = obs[history:]
    model_dyn = model[history:]
    fcst = np.zeros_like(obs_dyn)

    # Perform an initial training of the model
    kf.train_me(obs_train, model_train)

    for ij in range(len(obs_dyn)):
        # Provide a correction to the forecast
        fcst[ij] = kf.adjust_forecast(model_dyn[ij])

        # Update filter
        kf.train_me([obs_dyn[ij]], [model_dyn[ij]])

    fig = plt.figure(figsize=figSize)
    plt.plot(obs_dyn[pltRange], label='obs')
    plt.plot(model_dyn[pltRange], label='model')
    plt.plot(fcst[pltRange], label='kalman')
    plt.legend()
    plt.show()


    

    