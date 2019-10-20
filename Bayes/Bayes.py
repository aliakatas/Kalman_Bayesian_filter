"""
Defining the basic Bayesian Inference Corrector.
Can be used as standalone tool as well as in
combination with the Kalman filter.
"""

import numpy as np

class Bayes:
    def __init__(self, history):
        if history < 1:
            raise ValueError('History length is too small')

        # Set the bare minimum info needed
        self.history = history                  # History length
        self.obsValues = np.zeros(history)      # Initialise observations storage
        self.modValues = np.zeros(history)      # Initialise model results storage

        self.correctionType = 'none'            # Initialise type of data to be handled/corrected (Normal Dist)
        self.avgObs = None                      # Average of values of observations from history (Normal Dist)
        self.varObs = None                      # Variance of values of observations from history (Normal Dist)
        self.varError = None                    # Variance of error between observations and model results (Normal Dist)
        self.varCorrection = None               # Variance value to be used for correction (Normal Dist)

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
        self.avgObs = np.mean(myObs)
        self.varObs = np.var(myObs)
        self.varError = np.var(myModel - myObs)
        self.varCorrection = 1. / ((1./self.varError) + (1./self.varObs))

        


