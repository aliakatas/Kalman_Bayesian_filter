import numpy as np
from matplotlib import pyplot as plt
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
    kf = Kalman(history, order)
    bs = Bayes(history, distType='norm')
    
    # Use the first #history of them for training
    obsTrain = obs[:history]
    modelTrain = model[:history]

    # The rest to be used dynamically
    obsDyn = obs[history:]
    modelDyn = model[history:]
    fcst = np.zeros_like(obsDyn)

    # Perform an initial training of the model
    kf.train_me(obsTrain, modelTrain)
    bs.trainMe(obsTrain, modelTrain)

    for ij in range(len(obsDyn)):
        # Provide a correction to the forecast
        temp = kf.adjust_forecast(modelDyn[ij])
        fcst[ij] = bs.adjust_forecast(temp)

        # Update hybrid system
        kf.train_me([obsDyn[ij]], [modelDyn[ij]])
        bs.trainMe([obsDyn[ij]], [modelDyn[ij]])

    fig = plt.figure(figsize=figSize)
    plt.plot(obsDyn[pltRange], label='obs')
    plt.plot(modelDyn[pltRange], label='model')
    plt.plot(fcst[pltRange], '*', label='hybrid')
    plt.legend()
    plt.show()

    print('Test ended.')