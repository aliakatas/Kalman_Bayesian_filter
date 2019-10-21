import numpy as np
from matplotlib import pyplot as plt
from HybridKB import HybridKB

if __name__ == '__main__':
    print('Start Hybrid testing...')

    # Define some generic values
    historyKalman = 7
    orderKalman = 2
    historyBayes = 50

    # Create some imaginary data
    obs = np.random.normal(15.4, 2.3, size = 2000)
    model = obs + np.random.normal(2.5, 2.5, size = len(obs))

    # Define the plot range and size of the figure
    pltRange = range(1000, 1045)
    figSize = (11,9)

    # Use the first #history of them for training
    obsTrain = obs[:historyBayes]
    modelTrain = model[:historyBayes]

    # The rest to be used dynamically
    obsDyn = obs[historyBayes:]
    modelDyn = model[historyBayes:]
    fcst = np.zeros_like(obsDyn)

    # Start Testing
    # Create an instance of the hybrid system
    hyb = HybridKB(historyKalman, orderKalman, historyBayes, distType='norm')
    
    # Perform an initial training of the model
    hyb.trainMe(obsTrain, modelTrain)
    
    for ij in range(len(obsDyn)):
        # Provide a correction to the forecast
        fcst[ij] = hyb.adjustForecast(modelDyn[ij])

        # Update hybrid system
        hyb.trainMe([obsDyn[ij]], [modelDyn[ij]])

    fig = plt.figure(figsize=figSize)
    plt.plot(obsDyn[pltRange], label='obs')
    plt.plot(modelDyn[pltRange], '--', label='model')
    plt.plot(fcst[pltRange], '*', label='hybrid')
    plt.legend()
    plt.show()

    print('Test ended.')