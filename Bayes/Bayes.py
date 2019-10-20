"""
Defining the basic Bayesian Inference Corrector.
Can be used as standalone tool as well as in
combination with the Kalman filter.
"""

import numpy as np
from scipy import stats
import scipy.integrate as integrate

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

class Bayes:
    def __init__(self, history):
        if history < 1:
            raise ValueError('History length is too small')

        # Set the bare minimum info needed
        self.history = history                  # History length
        self.obsValues = np.zeros(history)      # Initialise observations storage
        self.modValues = np.zeros(history)      # Initialise model results storage

        self.correctionType = 'none'            # Initialise type of data to be handled/corrected (Normal Dist)
        self.maxData = 0.0                      # Maximum value to be encountered in data
        self.minData = 0.0                      # Minimum value to be encountered in data
        self.integrInterv = 1000                # Number of intervals to consider when integrating numerically (Weibull or Lognormal)
        
        # Normal dist related characteristics
        self.avgObs = None                      # Average of values of observations from history (Normal Dist)
        self.varObs = None                      # Variance of values of observations from history (Normal Dist)
        self.varError = None                    # Variance of error between observations and model results (Normal and Lognormal Dist)
        self.varCorrection = None               # Variance value to be used for correction (Normal Dist)

        # Lognormal dist related characterisics
        self.mu = None                          # Mean value of observations from history (based on lognormal dist)
        self.sigma = None                       # Std dev value of observations from history (based on lognormal dist)
        
        # Weibull dist related characteristics


    def trainMe(self, obs, model, distrType = None):
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

        NN = len(myObs)

        # TODO
        # Could add a method to auto-detect distribution...
        if distrType is None:
            self.correctionType = 'something'       # temporary for development
        else:
            self.correctionType = distrType

        # Hardwire normal for now...
        for ij in range(NN):
            self.updateHistory(myObs[ij], myModel[ij])

        self.updateCoefficientsNormal()

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

    def updateCoefficientsLognormal(self):
        """
        Given the history data, update the coefficients
        for lognormal distribution corrections.
        """
        self.varError = np.var(self.modValues - self.obsValues)
        tempObs = self.obsValues[self.obsValues > 0.0]

        shape, loc, scale = stats.lognorm.fit(tempObs, floc=0)
        self.mu = np.log(scale)
        self.sigma = shape

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
        return valNumerator / valDenominator

    def correctValueWeibull(self, pred):
        """
        Provides correction for the prediction pred,
        based on internal (history) values. (Weibull dist)
        """
        pass



