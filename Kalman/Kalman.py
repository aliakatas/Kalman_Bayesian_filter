"""
Defining the basic Kalman filtered
adjusted for correcting meteorological
data (ie no explicit system dynamics).
"""

import numpy as np

class Kalman:
    def __init__(self, history, dim, F = None, P = None, exp_var = None):
        
        if history < 1 or dim < 1:
            raise ValueError('Improper value entered for history length and/or dimension of observation matrix...')
            
        if dim < 2:
            print('Caution! Low accuracy due to the size of the observation matrix.')

        # Set the bare minimum info
        self.history = history
        self.dim = dim
        self.sq_shape = (dim, dim)
        self.vec_shape = (dim, 1)
        self.nTrained = 0

        # Set the system matrix based on entries
        self.set_system_matrix(F)

        # Set the covariance matrix based on entries
        self.set_covariance_matrix(P)

        # Hope it's classic Kalman, but you never know
        self.classic = True
        self.exp_var = None
        if not (exp_var is None):
            print('Switching to Information Geometry Kalman filter...')
            print('Variance for data is provided explicitly as: {}'.format(exp_var))
            self.classic = False
            self.exp_var = exp_var
        # Most likely, it's not going to be used right away, but planning ahead

        # Initialise other relevant matrices
        self.X = np.zeros(self.vec_shape)    # State vector
        self.H = np.zeros(self.vec_shape)    # Observations matrix
        self.KG = np.zeros(self.vec_shape)   # Kalman gain 

        # Create variables to keep history data
        self.x_history = np.zeros((self.dim, self.history + 1))
        self.H_history = np.zeros((self.dim, self.history + 1))
        self.y_history = np.zeros((self.history + 1, 1))

        
    def set_system_matrix(self, ff):
        # Get the system matrix
        if ff is None:
            self.F = np.eye(self.dim)
        else:
            if ff.shape[0] == self.dim:
                self.F = np.array(ff)
            else:
                raise ValueError('Transition (system) matrix F has the wrong dimensions.')
        return

    def set_covariance_matrix(self, pp):
        # Get the covariance matrix
        if pp is None:
            self.P = 10.0 * np.ones(self.sq_shape)
        else:
            if isinstance(pp, list):
                if pp.shape[0] == self.dim:
                    self.P = np.array(pp)
                else:
                    raise ValueError('Covariance matrix P has the wrong dimensions.')
            else:
                self.P = pp * np.ones(self.sq_shape)
        return 
    
    def set_observations_matrix(self, val):
        # Get the observations matrix from 
        # current model's prediction
        for ij in range(self.dim):
            self.H[ij, 0] = val ** ij
        return
        
    def print_2D_mat(self, mat, txt):
        # Just show me the contents
        print('{}:\t \t{}'.format(txt, mat[0, :]))
        for ij in range(1,self.dim):
            print(' \t \t{}'.format(mat[ij, :]))
        return

    def print_1D_mat(self, mat, txt):
        # Just show me the contents
        print('{}:\t \t{}'.format(txt, mat[:, 0]))

    def update_x_history(self, xx):
        # Update the x-history
        self.x_history[:, 0:-1] = self.x_history[:, 1:]
        self.x_history[:, -1] = xx[:, 0]
        return

    def update_H_history(self, hh):
        # Update the H-history 
        self.H_history[:, 0:-1] = self.H_history[:, 1:]
        self.H_history[:, -1] = hh[:, 0]
        return

    def update_y_history(self, yy):
        # Update the y-history
        self.y_history[0:-1, 0] = self.y_history[1:, 0]
        self.y_history[-1, 0] = yy
        return

    def get_state_variance(self):
        # Calculate the variance 
        # from the x-history
        sx = np.zeros_like(self.X)
        for ij in range(self.history):
            sx[:, 0] += self.x_history[:, ij] - self.x_history[:, ij + 1]
        sx /= self.history

        Wt = np.zeros_like(self.P)
        for ij in range(self.history):
            temp = self.x_history[:, ij] - self.x_history[:, ij + 1] - sx 
            Wt += np.dot(temp, temp.T)
            
        Wt /= (self.history - 1.0)
        return Wt

    def get_observation_variance(self):
        # Calculate the variance
        # from the y-history
        sy = 0.0
        for ij in range(self.history):
            sy += self.y_history[ij, 0] - np.dot(self.H_history[:, ij].T, self.x_history[:, ij])
        sy /= self.history

        Vt = 0.0
        for ij in range(self.history):
            temp = self.y_history[ij, 0] - np.dot(self.H_history[:, ij].T, self.x_history[:, ij]) - sy
            Vt += temp ** 2

        Vt /= (self.history - 1.0)
        return Vt
        
    def predict(self):
        # Provide an optimal estimate
        # for State vector and Covariance
        # matrix
        W = self.get_state_variance()
        self.X_est = np.dot(self.F, self.X)
        self.P_est = np.dot(np.dot(self.F, self.P), self.F.T) + W
        return

    def update(self, y):
        # Provide an update of 
        # the filter's state
        # based on new input
        
        # Get the variance of the dataset
        if self.classic:
            vt = self.get_observation_variance()
        else:
            vt = self.exp_var

        # Calculate the value for denominator
        denom = np.dot(np.dot(self.H.T, self.P_est), self.H) + vt
        
        # Kalman Gain
        self.KG = np.dot(self.P_est, self.H) / denom

        # Get new state
        self.X = self.X + self.KG * (y - np.dot(self.H.T, self.X_est))

        # Get new covariance matrix
        self.P = np.dot(np.eye(self.dim) - np.dot(self.KG, self.H.T), self.P_est)

        # Update history data
        self.update_x_history(self.X)
        self.update_H_history(self.H)
        self.update_y_history(y)
        return 

    def train_me(self, obs, model, showProrgess = False):
        """
        Master method to control the initial 
        training of the filter.
        """
        # Ensure they are numpy arrays
        myobs = np.array(obs)
        mymodel = np.array(model)
        
        # Check if the dimensions match
        if myobs.shape != mymodel.shape:
            raise TypeError('Initial training set does not have conforming shapes.')
        
        NN = len(myobs)

        # Train it using all the available data
        for ih in range(NN):
            
            self.nTrained += 1
            if showProrgess:
                print('Training #{}'.format(self.nTrained))

            y = myobs[ih] - mymodel[ih]

            self.set_observations_matrix(mymodel[ih])

            self.predict()

            self.update(y)

            if showProrgess:
                self.dump_members()
                print('************************************ \n')
        return

    def adjust_forecast(self, val, buff = 20.0):
        """
        Method to provide an adjustment to 
        the forecast value based on current
        state of the filter.
        """
        prod = np.dot(self.H.T, self.X)
        ret = val + prod

        if ret <= 0.0:
            ret = 0.0
        elif abs(prod) >= buff:
            ret = val
        return ret

    def dump_members(self):
        """
        Defining the "print" method for 
        debugging and informative purposes.
        """
        print('--------------------------')
        print('     Kalman Instance      ')
        print('Classic? \t{}'.format(self.classic))
        print('History: \t{}'.format(self.history))
        print('Order:   \t{}'.format(self.dim))
        self.print_2D_mat(self.F, 'F')
        self.print_2D_mat(self.P, 'P')
        self.print_2D_mat(self.KG, 'KG')
        self.print_1D_mat(self.X, 'X')
        self.print_1D_mat(self.H, 'H')
        self.print_2D_mat(self.x_history, 'x-hist')
        self.print_1D_mat(self.y_history, 'y-hist')
        print('Trained: \t{}'.format(self.nTrained))
        print('--------------------------')

    





        





        

