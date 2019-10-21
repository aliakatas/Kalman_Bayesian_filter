"""
Defining the basic Bayesian Inference Corrector.
Can be used as standalone tool as well as in
combination with the Kalman filter.
"""

import numpy as np
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
    dist_results = []
    params = {}
    for dist_name in distNames:
        dist = getattr(stats, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        #print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    # print("Best fitting distribution: "+str(best_dist))
    # print("Best p value: "+ str(best_p))
    # print("Parameters for the best fit: "+ str(params[best_dist]))
    return best_dist, best_p, params[best_dist]

class Bayes:
    def __init__(self, history, distType = None):
        # Check for invalid length of history data
        if history < 1:
            raise ValueError('History length is too small')
        
        # Check if the distribution is supported
        if distType is None:
            self.correctionType = 'none'            # Initialise type of data to be handled/corrected (will be detected)
        else:
            if distType.lower() in distAllowed.keys():
                self.correctionType = distAllowed[distType.lower()] # Initialise type of data to be handled/corrected (user defined)
            else:
                raise ValueError('Distribution not currently implemented for the system.')

        # Set the bare minimum info needed
        self.history = history                  # History length
        self.obsValues = np.zeros(history)      # Initialise observations storage
        self.modValues = np.zeros(history)      # Initialise model results storage

        self.maxData = 0.0                      # Maximum value to be encountered in data
        self.minData = 0.0001                   # Minimum value to be encountered in data
        self.integrInterv = 1000                # Number of intervals to consider when integrating numerically (Weibull or Lognormal)
        
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
            raise TypeError('Initial training set does not have conforming shapes.')

        # Update object's database
        NN = len(myObs)
        for ij in range(NN):
            self.updateHistory(myObs[ij], myModel[ij])

        # Check if the user would like to find best fit each time...
        if retarget:
           self.correctionType == 'none'

        # Check what distribution we are dealing with
        if self.correctionType == 'none':
            # Must detect it...
            findBestDistribution(myObs)
        
        # Perform update of coefficients
        if self.correctionType == 'norm':
            self.updateCoefficientsNormal()
        elif self.correctionType == 'lognorm':
            self.updateCoefficientsLognormal()
        elif self.correctionType == 'weibull_min':
            self.updateCoefficientsWeibull()

        # Update the maximum data value (estimate)
        mx = max(myObs)
        self.maxData = max([self.maxData, 10.0 * mx])
            
    def updateHistory(self, obs, model):
        """
        Update values stored as history data.
        """
        self.obsValues[1:] = self.obsValues[0:-1]
        self.modValues[1:] = self.modValues[0:-1]

        self.obsValues[0] = obs
        self.modValues[0] = model

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

    def adjust_forecast(self, model):
        """
        Method to control the correction
        of the forecast value.
        """
        if self.correctionType == 'norm':
            return self.correctValueNormal(model)
        elif self.correctionType == 'lognorm':
            return self.correctValueLognormal(model)
        elif self.correctionType == 'weibull_min':
            return self.correctValueWeibull(model)
        else:
            print('Requested dist: {}'.format(self.correctionType))
            raise TypeError('Unknown distribution type...')



