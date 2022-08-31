import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fmin_bfgs
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from numpy.linalg import det
from scipy.linalg import solve_triangular
import PyNomad
import sys

class StandardNormalization():
    def __init__(self, tab):
        self.tab = tab
        self.mean = np.mean(self.tab)
        self.std = np.std(self.tab)
    
    def transform(self, tab):
        return (tab - self.mean) / self.std
    
    def inv_transform(self, scaledTab):
        return (scaledTab * self.std) + self.mean


class MinMaxNormalization():
    def __init__(self, tab):
        self.tab = tab
        self.min = np.min(self.tab)
        self.max = np.max(self.tab)
    
    def transform(self, tab):
        return (tab - self.min) / (self.max - self.min)
    
    def inv_transform(self, scaledTab):
        return scaledTab * (self.max - self.min) + self.min

        

def plotGP(X_train, y_train, X_test, y_test, y_pred, variance, figureName="fig"):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(X_test, y_test, 'r--', linewidth=2, label='Test Function')
    ax.plot(X_train, y_train, 'ro', markerfacecolor='r', markersize=8, label='Training Data')
    ax.plot(X_test, y_pred, 'b-', linewidth=2, label='GP Prediction')
    ax.fill_between(X_test.flatten(), 
                    y_pred - 1.96 * np.sqrt(variance), 
                    y_pred + 1.96 * np.sqrt(variance),
                    facecolor='lightgrey',
                    label='95% Credibility Interval')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('f(x)', fontsize=15)
    ax.legend(loc="best",prop={'size': 9})
    plt.savefig('./images/' + figureName)
    #plt.show()


# RBF kernel
def squaredExponentialKernel(arg1, arg2, params=(1,1)):
    sigma, length = params[0], params[1]
    return sigma**2 * np.exp(-np.linalg.norm(arg1 - arg2)**2 / (2*length**2)) 


# Gaussian process
class GP:

    def __init__(self, covarianceFunction=squaredExponentialKernel):
        """
            Initialize a Gaussian Process model
            n_restarts: number of restarts of the local optimizer
            
        """  
        self.covFunction = covarianceFunction
        self.X = None 
        self.y = None 
        #self.theta = (1, 1, 10**-1)

           
    def calculateCovarianceMatrix(self, X1, X2, params):
        """
        Construct the correlation matrix between X1 and X2
        """
        return np.array([[self.covFunction(x1, x2, params) for x2 in X2] for x1 in X1])

    
    def negLikelihood(self, theta):
        self.theta = theta
        n = self.X.shape[0] # Number of training instances

        # # Construct correlation matrix between X_train and X_train
        # K = self.calculateCovarianceMatrix(self.X, self.X, self.theta) 
        # L = np.linalg.cholesky(K + self.theta[-1]**2 * np.eye(n))
        # alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        # # log marginal likelihood
        # logLikelihood = (-1/2 * np.dot((self.y).T, alpha)) - np.sum(np.log(np.diag(L))) - (n/2 * np.log(2 * np.pi))
        # return -logLikelihood.flatten()  # We return the negative of the computed likelihood value. This step is necessary since we will minimize the negative likelihood value 
        # # It is equivalent to maximizing the original likelihood value

        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        y = self.y.ravel()
        K = self.calculateCovarianceMatrix(self.X, self.X, self.theta) 
        L = np.linalg.cholesky(K + self.theta[-1]**2 * np.eye(n))
        S1 = solve_triangular(L, y, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        return np.sum(np.log(np.diagonal(L))) + 0.5 * np.dot(y, S2) + 0.5 * n * np.log(2*np.pi)
    

    def gradLogLikelihood(self, theta):
        self.theta = theta
        dlogtheta = np.zeros(3)
        n = self.X.shape[0] 
        
        #Construct correlation matrix between X_train and X_train
        K = self.calculateCovarianceMatrix(self.X, self.X, self.theta) 
        L = np.linalg.cholesky(K + (self.theta[-1]**2 + 1e-8) * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))

        sigma, length, sigmaN = self.theta[0], self.theta[1], self.theta[2]
        derivKsigma = np.array([[2*sigma * np.exp(-np.linalg.norm(x1 - x2) / (2*length**2)) for x2 in self.X] for x1 in self.X])
        derivKlength = np.array([[sigma**2 * np.exp(-np.linalg.norm(x1 - x2) / (2*length**2)) * (np.linalg.norm(x1 - x2) / (length**3)) for x2 in self.X] for x1 in self.X])
        alphaAlphaKinv = np.dot(alpha, alpha.T) - np.linalg.inv(K)
        dlogtheta[0] = 1/2 * np.trace(np.dot(alphaAlphaKinv, derivKsigma))
        dlogtheta[1] = 1/2 * np.trace(np.dot(alphaAlphaKinv, derivKlength))
        dlogtheta[2] = 1/2 * np.trace(np.dot(alphaAlphaKinv, 2 * sigmaN * np.eye(n)))
        return -dlogtheta



    def bbPynomad(self, x):
        try:
            f = self.negLikelihood([x.get_coord(0), x.get_coord(1), x.get_coord(2)])
            x.setBBO(str(f).encode("UTF-8"))
        except:
            print("Unexpected eval error", sys.exc_info()[0])
            return 0
        return 1  # 1: success 0: failed evaluation

    
    def nofit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.theta = (1, 1, 10**-1) # the signal variance (simga_f), length scale (l) and noise variance (sigma_n)
    

    def fit(self, X_train, y_train, n_startingPoints=20):

        self.X = X_train
        self.y = y_train
        
        # Normalize y
        scal_y = StandardNormalization(self.y)
        self.y = scal_y.transform(self.y)
        # Normalize X
        scal_X = MinMaxNormalization(self.X)
        self.X = scal_X.transform(self.X)
 
        # Bounds for the signal variance (sigma_f), length scale (l) and noise variance (sigma_n)
        bounds = [(1e-4, 10), (1e-2, 1e1), (1e-4, 1)]
        #bounds = [(1e-5, np.inf), (1e-5, np.inf), (1e-5, np.inf)]
        n_params = 3

        # best_theta, bestValue = None, np.inf
        # i = 0
        # # Random search method --------------------------------------
        # for p1 in np.random.uniform(bounds[0][0], bounds[0][1], 20):
        #     #print(i)
        #     i += 1
        #     for p2 in np.random.uniform(bounds[1][0], bounds[1][1], 20):
        #         for p3 in np.random.uniform(bounds[2][0], bounds[2][1], 20):
        #             self.theta = (p1, p2, p3)
        #             res = self.negLikelihood(self.theta)
        #             if res < bestValue:
        #                 best_theta = self.theta
        #                 bestValue = res
        # self.theta = best_theta

        # best_theta, bestValue = None, np.inf
        # i = 0
        # #Grid search method --------------------------------------
        # for p1 in np.linspace(bounds[0][0], bounds[0][1], 50):
        #     print(i)
        #     i += 1
        #     for p2 in np.linspace(bounds[1][0], bounds[1][1], 50):
        #         for p3 in np.linspace(bounds[2][0], bounds[2][1], 50):
        #             self.theta = (p1, p2, p3)
        #             res = self.negLikelihood(self.theta)
        #             if res < bestValue:
        #                 best_theta = self.theta
        #                 bestValue = res
        # self.theta = best_theta

        #Generate random starting points (Latin Hypercube)
        lhs = LatinHypercube(n_params).random(n_startingPoints)
        # Scale random samples to the given bounds 
        initial_points = np.zeros(lhs.shape)
        for i in range(n_params):
            lb, ub = bounds[i]
            initial_points[:,i] = (ub-lb) * lhs[:,i] + lb
        # Run local optimizer on all points
        opt_para = np.zeros((n_startingPoints, n_params)) # Record best parameters found for each starting points
        opt_func = np.zeros((n_startingPoints, 1)) # Record best value found for each starting points
        for i in range(n_startingPoints):
            res = minimize(self.negLikelihood, x0=initial_points[i,:], bounds=bounds, method='L-BFGS-B', jac=self.gradLogLikelihood)
            #res = minimize(self.negLikelihood, x0=initial_points[i,:], bounds=bounds, jac=self.gradLogLikelihood)
            opt_para[i,:] = res.x
            opt_func[i,:] = res.fun
        # Locate the optimum results and update theta
        self.theta = opt_para[np.argmin(opt_func)]
        #print(opt_para, opt_func)

        ## PYNOMAD ##
        # lb = [elt[0] for elt in bounds]
        # ub = [elt[1] for elt in bounds]
        # # Formatting the parameters for PyNomad
        # input_type = "BB_INPUT_TYPE (R R R)"
        # dimension = "DIMENSION 3"
        # max_nb_of_evaluations = "MAX_BB_EVAL 100"
        # params = [max_nb_of_evaluations, dimension, input_type,
        #         "DISPLAY_DEGREE 2", "BB_OUTPUT_TYPE OBJ", "DISPLAY_ALL_EVAL FALSE", "DISPLAY_STATS BBE OBJ (SOL)", "LH_SEARCH 25 0", "VNS_MADS_SEARCH TRUE"]
        # PyNomad.optimize(self.bbPynomad, [], lb, ub, params)
        # print()

        print("Best theta:" + str(self.theta))

        self.X, self.y = scal_X.inv_transform(self.X), scal_y.inv_transform(self.y)
    

    def updatePosterior(self, xNew, f):
        self.X = np.concatenate((self.X, xNew))
        self.y = np.concatenate((self.y, f(xNew)))
        self.fit(self.X, self.y)
        #self.nofit(self.X, self.y)


    def predict(self, X_test):
        """
        GP model predicting
        """

        # Normalize y
        scaler_y = StandardNormalization(self.y)
        self.y = scaler_y.transform(self.y)
        # Normalize X
        scaler_X = MinMaxNormalization(self.X)
        self.X = scaler_X.transform(self.X)
        #Normalize X_test
        X_test = scaler_X.transform(X_test)

        n = self.X.shape[0]
        # Construct correlation matrix betwenn X_train and X_train
        K = self.calculateCovarianceMatrix(self.X, self.X, self.theta)
        # Correlation matrix between X_train and X_test
        K_star = self.calculateCovarianceMatrix(self.X, X_test, self.theta)
        # Correlation matrix between X_test and X_test
        K_2star = self.calculateCovarianceMatrix(X_test, X_test, self.theta)

        L = np.linalg.cholesky(K + (self.theta[-1]**2 + 1e-8) * np.eye(n))
        # Mean prediction
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        mu_pred = np.dot(K_star.T, alpha)
      
        # Variance prediction
        v = np.linalg.solve(L, K_star)
        sigma_pred = K_2star - np.dot(v.T, v)
        variance = np.diag(sigma_pred)

        # Inverse transfromation
        self.X, self.y, X_test = scaler_X.inv_transform(self.X), scaler_y.inv_transform(self.y), scaler_X.inv_transform(X_test)

        mu_pred = scaler_y.inv_transform(mu_pred)
        #mu_pred = mu_pred*scaler_y.std + scaler_y.mean
        variance = variance * scaler_y.std

        return mu_pred.flatten(), sigma_pred, variance

        
    def score(self, X_test, y_test):
        """Calculate root mean squared error
        Input
        -----
        X_test: test set, array of shape (n_samples, n_features)
        y_test: test labels, array of shape (n_samples, )
        
        Output
        ------
        RMSE: the root mean square error"""
        y_pred, _, _ = self.predict(X_test)
        RMSE = np.sqrt(np.mean((y_pred - y_test)**2))
        return RMSE




### MAIN ###
if __name__ == '__main__':
    
    # Define the true function that we want to regress on
    #f = lambda x: (x)        
    f = lambda x: (x * np.sin(x))    
    #f = lambda x: (x + x**3)
    #f = lambda x: 100*(x**2 * np.exp(-x**2))
    #f = lambda x: (x*6-2)**2*np.sin(x*12-4)

    domain = (-2, 8)
    n1 = 3 # Number of points to condition on (training points)
    n2 = 100  # Number of points in posterior (test points)
    
    # Training data
    X_train = np.random.uniform(domain[0], domain[1], size=(n1,1))
    #X_train = np.linspace(domain[0], domain[1], n1).reshape(-1,1)
    y_train = f(X_train)

    # Testing data
    X_test = np.linspace(domain[0], domain[1], n2).reshape(-1,1)
    y_test = f(X_test)

    # GP model training
    gp = GP()
    gp.fit(X_train, y_train)
    #gp.nofit(X_train, y_train)
    
    # GP model predicting
    y_pred, sigma_pred, variance = gp.predict(X_test)

    # Plot
    plotGP(X_train, y_train, X_test, y_test, y_pred, variance, figureName="before")

    # Test add points

    for i in range(3):
        xNew = np.random.uniform(domain[0], domain[1], size=(1,1))
        gp.updatePosterior(xNew, f)
        y_pred, sigma_pred, variance = gp.predict(X_test)
        #print("RMSE: " + str(gp.score(X_test, y_test)))
        # Plot
        plotGP(gp.X, gp.y, X_test, y_test, y_pred, variance, figureName="after")

    plt.show()
   

