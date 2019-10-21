import numpy as np
from scipy.stats import weibull_min
from Bayes import Bayes
from matplotlib import pyplot as plt

#testDists = ['norm', 'weibull_min', 'lognorm']
testDists = ['weibull_min']

def testNormal(hLen, totN, figSize, pltRange):
    """
    Brief test with data following normal distribution.
    """
    # Create some imaginary data
    obs = np.random.normal(15.4, 1.3, size = totN)
    model = obs + np.random.normal(0.0, 4.5, size = len(obs))

    # Start Testing
    # Create an instance of the object
    bs = Bayes(hLen, distType = 'norm')

    # Use the first #hLen of them for training
    obsTrain = obs[:hLen]
    modelTrain = model[:hLen]

    # The rest to be used dynamically
    obsDyn = obs[hLen:]
    modelDyn = model[hLen:]
    fcst = np.zeros_like(obsDyn)

    # Perform an initial training of the model
    bs.trainMe(obsTrain, modelTrain)

    for ij in range(len(obsDyn)):
        # Provide a correction to the forecast
        fcst[ij] = bs.adjust_forecast(modelDyn[ij])

        # Update system
        bs.trainMe([obsDyn[ij]], [modelDyn[ij]])
    
    plt.figure(figsize = figSize)
    plt.plot(obsDyn[pltRange], label='obs')
    plt.plot(modelDyn[pltRange], label='model')
    plt.plot(fcst[pltRange], '*', label='Bayes')
    plt.title('Normal Dist Data')
    plt.legend()
    plt.show()
    return

def testWeibull(hLen, totN, figSize, pltRange):
    """
    Brief test with data following weibull distribution.
    """
    # Create some imaginary data
    obs = weibull_min.rvs(c = 1.96, scale = 2.1, size = totN) # c: shape
    model = obs + np.random.normal(0.0, 2.5, size = len(obs))

    # Start Testing
    # Create an instance of the object
    bs = Bayes(hLen, distType = 'weibull_min')

    # Use the first #hLen of them for training
    obsTrain = obs[:hLen]
    modelTrain = model[:hLen]

    # The rest to be used dynamically
    obsDyn = obs[hLen:]
    modelDyn = model[hLen:]
    fcst = np.zeros_like(obsDyn)

    # Perform an initial training of the model
    bs.trainMe(obsTrain, modelTrain)

    for ij in range(len(obsDyn)):
        # Provide a correction to the forecast
        fcst[ij] = bs.adjust_forecast(modelDyn[ij])

        # Update system
        bs.trainMe([obsDyn[ij]], [modelDyn[ij]])

    plt.figure(figsize = figSize)
    plt.plot(obsDyn[pltRange], label='obs')
    plt.plot(modelDyn[pltRange], label='model')
    plt.plot(fcst[pltRange], '*', label='Bayes')
    plt.title('Weibull Dist Data')
    plt.legend()
    plt.show()
    return

def testLognormal():
    """
    Brief test with data following lognormal distribution.
    """
    pass

if __name__ == '__main__':

    # Define some generic values
    history = 100
    totalNum = 2000
    plotNum = 40

    # Define the plot range and size of the figure
    figSize = (11, 9)
    pIn = int(totalNum * .5)
    pltRange = range(pIn, pIn + plotNum)
    
    for item in testDists:
        if item.lower() == 'norm':
            print('\n *****   Normal Distribution data testing    ***** ')
            testNormal(history, totalNum, figSize, pltRange)
            print('\n ***** Normal Distribution data testing  END ***** ')
        elif item.lower() == 'weibull_min':
            print('\n *****   Weibull Distribution data testing    ***** ')
            testWeibull(history, totalNum, figSize, pltRange)
            print('\n ***** Weibull Distribution data testing  END ***** ')
        elif item.lower() == 'lognorm':
            print('\n *****   Lognormal Distribution data testing    ***** ')
            testLognormal()
            print('\n ***** Lognormal Distribution data testing  END ***** ')
        else:
            raise TypeError('Unknown distribution used for testing...')

    
    


    