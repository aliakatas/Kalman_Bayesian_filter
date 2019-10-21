"""
Defining the basic Bayesian Inference Corrector.
Can be used as standalone tool as well as in
combination with the Kalman filter.
"""

import numpy as np
import math
from scipy import stats
import scipy.integrate as integrate
import scipy.stats as stats

# Names of distributions currently supported by the system 
distNames = ['norm', 'lognorm', 'weibull_min']
distAllowed = {
    'norm':'norm', 
    'lognorm':'lognorm', 
    'weibull_min':'weibull_min', 
    'normal':'norm', 
    'lognormal':'lognorm', 
    'log':'lognorm', 
    'weibull':'weibull_min', 
    'wei':'weibull_min', 
    'weib':'weibull_min'
}

def lognormalIntegralNumerator(x, model, varErr, mu, sigma):
    """
    Function used in the numerator of the integral
    performing the correction.
    """
    a1 = ((model - x) ** 2) / (2. * varErr ** 2)
    b1 = ((np.log(x) - mu) ** 2) / (2 * sigma ** 2)
    return np.exp(-a1 - b1)

def lognormalIntegralDenominator(x, model, varErr, mu, sigma):
    """
    Function used in the denominator of the integral
    performing the correction.
    """
    a1 = ((model - x) ** 2) / (2. * varErr ** 2)
    b1 = ((np.log(x) - mu) ** 2) / (2 * sigma ** 2)
    return np.exp(-a1 - b1) / x

def weibullIntegralNumerator(x, model, varErr, shape, scale):
    """
    Function used in the numerator of the integral
    performing the correction.
    """
    return (x ** shape) * np.exp(-(x/scale) ** shape) * np.exp((-(model - x) ** 2) / (2. * varErr))

def weibullIntegralDenominator(x, model, varErr, shape, scale):
    """
    Function used in the denominator of the integral
    performing the correction.
    """
    return (x ** (shape - 1.)) * np.exp(-(x/scale) ** shape) * np.exp((-(model - x) ** 2) / (2. * varErr))

def findBestDistribution(data):
    """
    Perform some tests to determine the distribution
    that best fits the data.
    """
    best_dist = 'none'
    best_p = -9999.
    paramOut = []
    for dist_name in distNames:
        dist = getattr(stats, dist_name)
        param = dist.fit(data)

        # Applying the Kolmogorov-Smirnov test
        _, p = stats.kstest(data, dist_name, args=param)
        #print("p value for "+dist_name+" = "+str(p))
        if not math.isnan(p):
            if p > best_p:
                best_dist = dist_name
                best_p = p
                paramOut = param

    return best_dist, best_p, paramOut

class Bayes:
    def __init__(self, history, distType = None):
        # Check for invalid length of history data
        if history < 1:
            print('Bayes module reporting: ')
            raise ValueError('History length is too small')
        
        # Check if the distribution is supported
        if distType is None:
            self.correctionType = 'none'            # Initialise type of data to be handled/corrected (will be detected)
        else:
            if distType.lower() in distAllowed.keys():
                self.correctionType = distAllowed[distType.lower()] # Initialise type of data to be handled/corrected (user defined)
            else:
                print('Bayes module reporting: ')
                raise ValueError('Distribution not currently implemented for the system.')

        # Set the bare minimum info needed
        self.history = history                  # History length
        self.obsValues = np.zeros(history)      # Initialise observations storage
        self.modValues = np.zeros(history)      # Initialise model results storage

        self.maxData = 0.0                      # Maximum value to be encountered in data
        self.minData = 0.0001                   # Minimum value to be encountered in data
        self.nTrained = 0                       # To count the number of times the object received training

        # Normal dist related characteristics
        self.avgObs = None                      # Average of values of observations from history (Normal Dist)
        self.varObs = None                      # Variance of values of observations from history (Normal Dist)
        self.varCorrection = None               # Variance value to be used for correction (Normal Dist)
        self.varError = None                    # Variance of error between observations and model results (Normal, Lognormal and Weibull Dists)

        # Lognormal dist related characterisics
        self.mu = None                          # Mean value of observations from history (based on Lognormal Dist)
        self.sigma = None                       # Std dev value of observations from history (based on Lognormal Dist)
        
        # Weibull dist related characteristics
        self.scale = None                       # Scale parameter value of observations from history (based on Weibull Dist) 
        self.shape = None                       # Shape parameter value of observations from history (based on Weibull Dist) 

    def trainMe(self, obs, model, retarget = False):
        """
        Master method to control the 
        initial training of the system.
        """
        # Ensure it's working with numpy arrays
        myObs = np.array(obs)
        myModel = np.array(model)

        # Check if shapes match
        if myObs.shape != myModel.shape:
            print('Bayes module reporting: ')
            raise TypeError('Initial training set does not have conforming shapes.')

        # Update object's database
        NN = len(myObs)
        if NN > self.history:
            print('Bayes module reporting: ')
            print('WARNING: Dimensions of training set exceeds length of history database.')
        for ij in range(NN):
            self.updateHistory(myObs[ij], myModel[ij])
            self.nTrained += 1

        # Check if the user would like to find best fit each time...
        if retarget:
           self.correctionType = 'none'

        # Check what distribution we are dealing with
        if self.correctionType == 'none':
            # Must detect it...
            best_dist, best_p, params = findBestDistribution(myObs)

            if best_dist != 'none':
                self.correctionType = best_dist
        
        # Perform update of coefficients
        if self.correctionType == 'norm':
            self.updateCoefficientsNormal()
        elif self.correctionType == 'lognorm':
            self.updateCoefficientsLognormal()
        elif self.correctionType == 'weibull_min':
            self.updateCoefficientsWeibull()

        # Update the maximum data value (estimate)
        mx = np.nanmax(myObs)
        self.maxData = np.nanmax([self.maxData, 2.5 * mx])
            
    def updateHistory(self, obs, model):
        """
        Update values stored as history data.
        """
        self.obsValues[0:-1] = self.obsValues[1:]
        self.modValues[0:-1] = self.modValues[1:]

        self.obsValues[-1] = obs
        self.modValues[-1] = model

    def updateCoefficientsNormal(self):
        """
        Given the history data, update the coefficients
        for normal distribution corrections.
        """
        self.avgObs = np.mean(self.obsValues)
        self.varObs = np.var(self.obsValues)
        self.varError = np.var(self.modValues - self.obsValues)
        self.varCorrection = 1. / ((1./self.varError) + (1./self.varObs))
        return 

    def updateCoefficientsLognormal(self):
        """
        Given the history data, update the coefficients
        for lognormal distribution corrections.
        """
        self.varError = np.var(self.modValues - self.obsValues)
        tempObs = self.obsValues[self.obsValues > 0.0]

        shape, _, scale = stats.lognorm.fit(tempObs, floc=0)
        self.mu = np.log(scale)
        self.sigma = shape
        return 
    
    def updateCoefficientsWeibull(self):
        """
        Given the history data, update the coefficients
        for weibull distribution corrections.
        """
        self.varError = np.var(self.modValues - self.obsValues)
        tempObs = self.obsValues[self.obsValues > 0.0]

        shape, _, scale = stats.weibull_min.fit(tempObs, loc = 0)
        self.shape = shape 
        self.scale = scale
        return 

    def correctValueNormal(self, pred):
        """
        Provides correction for the prediction pred,
        based on internal (history) values. (Normal dist)
        """
        return ((1./self.varError) * pred + (1./self.varObs) * self.avgObs) * self.varCorrection

    def correctValueLognormal(self, pred):
        """
        Provides correction for the prediction pred,
        based on internal (history) values. (Lognormal dist)
        """
        valNumerator = integrate.quad(lambda x: lognormalIntegralNumerator(x, pred, self.varError, self.mu, self.sigma), self.minData, self.maxData)
        valDenominator = integrate.quad(lambda x: lognormalIntegralDenominator(x, pred, self.varError, self.mu, self.sigma), self.minData, self.maxData)
        return valNumerator[0] / valDenominator[0]

    def correctValueWeibull(self, pred):
        """
        Provides correction for the prediction pred,
        based on internal (history) values. (Weibull dist)
        """
        valNumerator = integrate.quad(lambda x: weibullIntegralNumerator(x, pred, self.varError, self.shape, self.scale), self.minData, self.maxData)
        valDenominator = integrate.quad(lambda x: weibullIntegralDenominator(x, pred, self.varError, self.shape, self.scale), self.minData, self.maxData)
        return valNumerator[0] / valDenominator[0]

    def adjustForecast(self, model , buff = 20.0):
        """
        Method to control the correction
        of the forecast value.
        """
        ret = model
        if self.correctionType == 'norm':
            ret = self.correctValueNormal(model)
        elif self.correctionType == 'lognorm':
            ret = self.correctValueLognormal(model)
        elif self.correctionType == 'weibull_min':
            ret = self.correctValueWeibull(model)
        else:
            print('Requested dist: {}'.format(self.correctionType))
            raise TypeError('Unknown distribution type...')
        
        if abs(model - ret) > buff:
            ret = model

        return ret

    def dumpMembers(self):
        """
        Defining the "print" method for 
        debugging and informative purposes.
        """
        print('--------------------------')
        print('      Bayes Instance      ')
        print('Type?        \t{}'.format(self.correctionType))
        print('History:     \t{}'.format(self.history))
        if self.correctionType == 'norm':
            print('Obs avg:     \t{}'.format(self.avgObs))
            print('Obs var:     \t{}'.format(self.varObs))
            print('Error var:   \t{}'.format(self.varError))
        elif self.correctionType == 'lognorm':
            print('Obs mean:    \t{}'.format(self.mu))
            print('Obs sigma:   \t{}'.format(self.sigma))
        elif self.correctionType == 'weibull_min':
            print('Obs shape:   \t{}'.format(self.shape))
            print('Obs scale:   \t{}'.format(self.scale))
        else:
            print('Unknown distribution...')
            print('No parameters to show.')

        print('Trained: \t{}'.format(self.nTrained))
        print('--------------------------')

