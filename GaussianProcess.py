import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler



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
def squaredExponentialKernel(arg1, arg2, params=(1,1,10**-8)):
    sigma, length, noise = params[0], params[1], params[2]
    #deltakro = 1 if arg1 == arg2 else 0
    return sigma**2 * np.exp(-np.linalg.norm(arg1 - arg2)**2 / (2*length**2)) #+ deltakro * noise


# Gaussian process
class GP:

    def __init__(self, covarianceFunction=squaredExponentialKernel):
        """
            Initialize a Gaussian Process model
            n_restarts: number of restarts of the local optimizer
            
        """  
        self.covFunction = covarianceFunction
        self.X = None # X_train will be initialized in the fit
        self.y = None # y_train will be initialized in the fit

           
    def calculateCovarianceMatrix(self, X1, X2, params):
        """
        Construct the correlation matrix between X1 and X2
        """
        # signalSigma, length, noiseVariance = params[0], params[1], params[2]
        # K = np.zeros((X1.shape[0], X2.shape[0]))
        # for i in range(X1.shape[0]):
        #     for j in range(X2.shape[0]):
        #         K[i,j] = signalSigma**2 * np.exp(-np.square(abs(X1[i] - X2[j])) / (2*length**2)) #+ kron * noiseVariance**2
        # return K
        return np.array([[self.covFunction(x1, x2, params) for x2 in X2] for x1 in X1])

    
    def negLikelihood(self, theta):

        self.theta = theta

        n = self.X.shape[0] # Number of training instances

        # Construct correlation matrix betwenn X_train and X_train
        K = self.calculateCovarianceMatrix(self.X, self.X, self.theta) 
        # print(K)
        # print()
        try:
            L = np.linalg.cholesky(K + self.theta[-1] * np.eye(n))
        except Exception as e:
            print(e)
            print(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))

        # log marginal likelihood
        logLikelihood = (-1/2 * np.dot((self.y).T, alpha)) - np.sum(np.log(np.diag(L))) - (n/2 * np.log(2 * np.pi))
        
        #Ky = K + self.theta[-1] * np.eye(n)
        #logLikelihood = -1/2 * np.dot((self.y).T, np.dot(np.linalg.inv(Ky), self.y)) - 1/2 * np.log(np.linalg.det(Ky)) - (n/2 * np.log(2 * np.pi))
        #logLikelihood = -n/2 * np.log(1/n * np.dot((self.y).T, np.dot(np.linalg.inv(K+ self.theta[-1] * np.eye(n)), self.y))) - 1/2 * np.log(np.linalg.det(K+ self.theta[-1] * np.eye(n))) - (n/2 * np.log(2 * np.pi))
        # variance = 1/n * np.dot((self.y).T, np.dot(np.linalg.inv(K), self.y))
        #print(variance)

        return -logLikelihood.flatten()  # We return the negative of the computed likelihood value. This step is necessary since we will minimize the negative likelihood value 
        # It is equivalent to maximizing the original likelihood value

    
    def nofit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.theta = (1, 1, 10**-1) # the signal variance (simga_f), length scale (l) and noise variance (sigma_n)
    

    def fit(self, X_train, y_train, n_startingPoints=20):

        self.X = X_train
        self.y = y_train
        
        # Normalize y
        # self.y_mean = self.y.mean()
        # self.y_std = self.y.std()
        # self.y = (self.y - self.y_mean)/ self.y_std
        # self.scalerY = StandardScaler()
        # self.scalerY.fit(self.y)
        # self.y = self.scalerY.transform(self.y)
        

        # Normalize x
        # self.X_mean = self.y.mean()
        # self.X_std = self.y.std()
        # self.X = (self.X - self.X_mean)/ self.X_std
        # self.scalerX = StandardScaler()
        # self.scalerX.fit(self.X)
        # self.X = self.scalerX.transform(self.X)
        # self.X_mean = []
        # self.X_std = []
        # for d in range(self.X.shape[1]):
        #     self.X_mean.append(self.X[:, d].mean())
        #     self.X_std.append(self.X[:, d].std())
        #     self.X[:, d] = (self.X[:, d] - self.X_mean[d]) / self.X_std[d]
 
        # Bounds for the signal variance (sigma_f), length scale (l) and noise variance (sigma_n)
        bounds = [(1e-4, 10), (1e-2, 1e1), (1e-4, 1)]
        n_params = 3

        best_theta, bestValue = None, np.inf
        i = 0
        # Random search method --------------------------------------
        for p1 in np.random.uniform(bounds[0][0], bounds[0][1], 20):
            #print(i)
            i += 1
            for p2 in np.random.uniform(bounds[1][0], bounds[1][1], 20):
                for p3 in np.random.uniform(bounds[2][0], bounds[2][1], 20):
                    self.theta = (p1, p2, p3)
                    res = self.negLikelihood(self.theta)
                    if res < bestValue:
                        best_theta = self.theta
                        bestValue = res
        self.theta = best_theta

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

        # # Generate random starting points (Latin Hypercube)
        # lhs = LatinHypercube(n_params).random(n_startingPoints)
        # # Scale random samples to the given bounds 
        # initial_points = np.zeros(lhs.shape)
        # for i in range(n_params):
        #     lb, ub = bounds[i]
        #     initial_points[:,i] = (ub-lb) * lhs[:,i] + lb
        # # Run local optimizer on all points
        # opt_para = np.zeros((n_startingPoints, n_params)) # Record best parameters found for each starting points
        # opt_func = np.zeros((n_startingPoints, 1)) # Record best value found for each starting points
        # for i in range(n_startingPoints):
        #     res = minimize(self.negLikelihood, x0=initial_points[i,:], bounds=bounds)#, method='BFGS')
        #     opt_para[i,:] = res.x
        #     opt_func[i,:] = res.fun
        # # Locate the optimum results and update theta
        # self.theta = opt_para[np.argmin(opt_func)]
        # #print(opt_para, opt_func)

        print("Best theta:" + str(self.theta))
    

    def updatePosterior(self, xNew, f):
        self.X = np.concatenate((self.X, xNew))
        self.y = np.concatenate((self.y, f(xNew)))
        self.fit(self.X, self.y)
        #self.nofit(self.X, self.y)


    def predict(self, X_test):
        """
        GP model predicting
        """

        # Normalize X_test
        # for d in range(X_test.shape[1]):
        #     X_test[:, d] = (X_test[:, d] - X_test[:, d].mean()) / X_test[:, d].std()
        #X_test = (X_test - X_test.mean())/ X_test.std()
        # self.scalerX_test = StandardScaler()
        # self.scalerX_test.fit(X_test)
        # X_test = self.scalerX_test.transform(X_test)

        
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

        # Convert y_pred back to true scale
        #mu_pred = self.scalerY.inverse_transform(mu_pred)
        #mu_pred = mu_pred * self.y_std + self.y_mean
        #variance = variance * self.scalerY.var_

        # Convert X, y
        # self.X = self.scalerX.inverse_transform(self.X)
        # self.y = self.scalerY.inverse_transform(self.y)
        # for d in range(self.X.shape[1]):
        #     self.X[:, d] = (self.X[:, d] * self.X_std[d] + self.X_mean[d])
    
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

    domain = (0, 3)
    n1 = 3 # Number of points to condition on (training points)
    n2 = 100  # Number of points in posterior (test points)
    
    # Training data
    #X_train = np.random.uniform(domain[0], domain[1], size=(n1,1))
    X_train = np.linspace(domain[0], domain[1], n1).reshape(-1,1)
    y_train = f(X_train)

    # Testing data
    X_test = np.linspace(domain[0], domain[1], n2).reshape(-1,1)
    # etre sur qu'on a pas de x_train dedans
    y_test = f(X_test)


    # GP model training
    gp = GP()
    gp.fit(X_train, y_train)
    #gp.nofit(X_train, y_train)


    # GP model predicting
    y_pred, sigma_pred, variance = gp.predict(X_test)


    # # Plot
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
   



    