import numpy as np
from Kalman import Kalman
from Bayes import Bayes

if __name__ == '__main__':
    print('Start Hybrid testing...')

    # Define some generic values
    history = 7
    order = 2

    # Create some imaginary data
    obs = np.random.normal(15.4, 2.3, size = 2000)
    model = obs + np.random.normal(2.5, 2.5, size = len(obs))

    # Define the plot range and size of the figure
    pltRange = range(1000, 1045)
    figSize = (11,9)

    # Start Testing
    # Create an instance of the hybrid system
    kf = Kalman.(history, order)
    bs = Bayes.(history, distType='norm')
    
    # Use the first #history of them for training
    obs_train = obs[:history]
    model_train = model[:history]

    # The rest to be used dynamically
    obs_dyn = obs[history:]
    model_dyn = model[history:]
    fcst = np.zeros_like(obs_dyn)

    # Perform an initial training of the model
    kf.train_me(obs_train, model_train)

    print('Test ended.')