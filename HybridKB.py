import numpy as np
from Kalman import Kalman
from Bayes import Bayes, distAllowed

class HybridKB(object):

    def __init__(self, historyKal, dimKal, historyBayes, distType = None):

        # First check Kalman related parameters
        if historyKal < 1 or dimKal < 1:
            print('Hybrid module reporting: ')
            raise ValueError('Improper value entered for history length and/or dimension of observation matrix (Kalman)...')
            
        if dimKal < 2:
            print('Hybrid module reporting: ')
            print('Caution! Low accuracy due to the size of the observation matrix. (Affecting mostly Kalman)')

        # Now check for Bayes related parameters
        if historyBayes < 1:
            print('Hybrid module reporting: ')
            raise ValueError('History length is too small (Bayes)...')

        # Final check to ensure smooth operations
        if historyBayes < historyKal:
            print('Hybrid module reporting: ')
            raise ValueError('History length for Bayesian system must be greater or equal to that of Kalman. ')
        
        # Check if the distribution is supported
        if distType is None:
            correctionType = 'none'            # Initialise type of data to be handled/corrected (will be detected)
        else:
            if distType.lower() in distAllowed.keys():
                correctionType = distAllowed[distType.lower()] # Initialise type of data to be handled/corrected (user defined)
            else:
                print('Hybrid module reporting: ')
                raise ValueError('Distribution not currently implemented for the system.')
        
        # Now ready to use the input (could defer to classes but thought better to do it here as well)
        self.KF = Kalman(historyKal, dimKal)
        self.BS = Bayes(historyBayes, correctionType)

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
            raise TypeError('Training set does not have conforming shapes.')
        
        # Train each module
        self.KF.trainMe(myObs, myModel)
        self.BS.trainMe(myObs, myModel, retarget = retarget)

    def adjustForecast(self, model, buff = 20.0):
        """
        Method to control the correction
        of the forecast value.
        """
        tempVal = self.KF.adjustForecast(model, buff = buff)
        ret = self.BS.adjustForecast(tempVal, buff = buff)

        return ret
        


    
