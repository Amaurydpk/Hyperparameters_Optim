import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube


def plotGP(X_train, y_train, X_test, y_test, y_pred, variance):
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
    plt.savefig('./images/Test_1D_results')
    plt.show()


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
        self.X = None # X_train will be initialized in the fit
        self.Y = None # y_train will be initialized in the fit

           
    def calculateCovarianceMatrix(self, X1, X2, params):
        """
        Construct the correlation matrix between X1 and X2
        """
        # signalVariance, length, noiseVariance = params[0], params[1], params[2]
        # K = np.zeros((X1.shape[0], X2.shape[0]))
        # for i in range(X1.shape[0]):
        #     for j in range(X2.shape[0]):
        #         #kron = 1 if i == j else 0
        #         K[i,j] = signalVariance**2 * np.exp(-np.square(abs(X1[i] - X2[j])) / (2*length**2)) #+ kron * noiseVariance**2
        # return K
        return np.array([[self.covFunction(x1, x2, params) for x2 in X2] for x1 in X1])

    
    def negLikelihood(self, theta):

        self.theta = theta

        n = self.X.shape[0] # Number of training instances

        # Construct correlation matrix betwenn X_train and X_train
        K = self.calculateCovarianceMatrix(self.X, self.X, self.theta) #+ (1e-10 + self.noise) * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K + self.theta[-1] * np.eye(n))
        except Exception as e:
            print(e)
            print(K)

      
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))

        # log marginal likelihood
        logLikelihood = (-1/2 * np.dot((self.y).T, alpha)) - np.sum(np.log(np.diag(L))) - (n/2 * np.log(2 * np.pi))

        #logLikelihood = -n/2 * np.log(1/n * np.dot((self.y).T, np.dot(np.linalg.inv(K), self.y))) - 1/2 * np.log(np.linalg.det(K)) - (n/2 * np.log(2 * np.pi))
        # variance = 1/n * np.dot((self.y).T, np.dot(np.linalg.inv(K), self.y))
        #print(variance)

        return -logLikelihood.flatten()  # We return the negative of the computed likelihood value. This step is necessary since we will minimize the negative likelihood value 
        # It is equivalent to maximizing the original likelihood value

    
    def nofit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.theta = (1, 1, 10**-6) # the signal variance (simga_f), length scale (l) and noise variance (sigma_n)
    

    def fit(self, X_train, y_train, n_startingPoints=20):

        self.X = X_train
        self.y = y_train
      
        # Bounds for the signal variance (simga_f), length scale (l) and noise variance (sigma_n)
        bounds = [(0, 10), (10**-2, 10**2), (10**-8, 10**-1)]
        n_params = 3

        #best_theta = self.theta
        #bestValue = np.inf
        # ## LHS method ----------------------------------------------
        # # Generate random starting points (Latin Hypercube)
        # lhs = LatinHypercube(n_params).random(n_startingPoints)
        # #  Scale random samples to the given bounds 
        # initial_points = (ub-lb) * lhs + lb
        # for point in initial_points:
        #     self.theta = (point[0], point[1])
        #     res = self.negLikelihood(self.theta)
        #     if res < bestValue:
        #         best_theta = self.theta
        #         bestValue = res

        # Grid search method --------------------------------------
        # param1 = np.linspace(sigmaLB, sigmaUB, 100)
        # param2 = np.linspace(lLB, lUB, 100)
        # for p1 in param1:
        #     for p2 in param2:
        #         self.theta = (p1, p2)
        #         res = self.negLikelihood(self.theta)
        #         if res < bestValue:
        #             best_theta = self.theta
        #             bestValue = res
        #self.theta = best_theta

        # Generate random starting points (Latin Hypercube)
        lhs = LatinHypercube(n_params).random(n_startingPoints)
   
        # Scale random samples to the given bounds 
        initial_points = np.zeros(lhs.shape)
        for i in range(n_params):
            lb, ub= bounds[i]
            initial_points[:,i] = (ub-lb) * lhs[:,i] + lb

        # Run local optimizer on all points
        opt_para = np.zeros((n_startingPoints, n_params)) # Record best parameters found for each starting points
        opt_func = np.zeros((n_startingPoints, 1)) # Record best value found for each starting points

        for i in range(n_startingPoints):
            res = minimize(self.negLikelihood, x0=initial_points[i,:], bounds=bounds)
            opt_para[i,:] = res.x
            opt_func[i,:] = res.fun
        # Locate the optimum results and update theta
        self.theta = opt_para[np.argmin(opt_func)]

        #print(opt_para, opt_func)
        print("Best theta:" + str(self.theta))

        
    def predict(self, X_test):
        """
        GP model predicting
        """
        n = self.X.shape[0]
        # Construct correlation matrix betwenn X_train and X_train
        K = self.calculateCovarianceMatrix(self.X, self.X, self.theta)
        # Correlation matrix between X_train and X_test
        K_star = self.calculateCovarianceMatrix(self.X, X_test, self.theta)
        # Correlation matrix between X_test and X_test
        K_2star = self.calculateCovarianceMatrix(X_test, X_test, self.theta)

        L = np.linalg.cholesky(K + self.theta[-1] * np.eye(n))
        # Mean prediction
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        mu_pred = np.dot(K_star.T, alpha)
      
        # Variance prediction
        v = np.linalg.solve(L, K_star)
        sigma_pred = K_2star - np.dot(v.T, v)
        variance = np.diag(sigma_pred)
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
    #f = lambda x: (x * np.sin(x))    
    #f = lambda x: (x + x**3)
    #f = lambda x: 100*(x**2 * np.exp(-x**2))
    f = lambda x: (x*6-2)**2*np.sin(x*12-4)

    domain = (1, 2)
    n1 = 5 # Number of points to condition on (training points)
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
    
    #print(variance)
    # print(np.max(variance))
    # print(np.mean( (y_pred - 1.96 * np.sqrt(variance)) - 
    #                (y_pred + 1.96 * np.sqrt(variance))
    #             )
    # )

    print("RMSE: " + str(gp.score(X_test, y_test)))

    # Plot
    plotGP(X_train, y_train, X_test, y_test, y_pred, variance)
   



    