import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from scipy.linalg import solve_triangular, cho_solve, cho_factor
import sys


def display2D(X1, X2, Z, X_train, y_train, title):
    X1, X2 = np.meshgrid(X1, X2)
    #Z = f(X1, X2)
    fig = plt.figure(figsize=plt.figaspect(0.3))
    fig.suptitle(title)
    # first subplot
    ax = fig.add_subplot(1, 2, 1)
    ax.contour(X1, X2, Z)
    ax.scatter(X_train[:,0], X_train[:,1], y_train, c="black", label="Training points")
    #ax.scatter(, c='r', label='Training Data')
    # second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # plot surface
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.scatter(X_train[:,0], X_train[:,1], y_train, c="black", s=200, label="Training points")
    # Customize the z axis
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_locator(FormatStrFormatter('%.02f')))
    # add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend()


def display2Dv2(X1, X2, Z_true, Z_pred, X_train, y_train):
    X1, X2 = np.meshgrid(X1, X2)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plot surface
    surf = ax.plot_surface(X1, X2, Z_true, linewidth=0, antialiased=True, label="Objective function") #alpha=0.5)
    surf = ax.plot_surface(X1, X2, Z_pred, linewidth=0, antialiased=True, label="Prediction")
    ax.scatter(X_train[:,0], X_train[:,1], y_train, c="black", s=100, label="Training points")
    # add a color bar which maps values to colors
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.legend()


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
    #plt.savefig('./images/' + figureName)
    #plt.show()


# RBF kernel
def squaredExponentialKernel(X1, X2, params=(1, 1)):

    # X1 (n1 x d) : Array of n1 points, where each point has dim d : array of line vectors
    # X2 (n2 x d) : Array of n2 points, where each point has dim d : array of line vectors

    # Higher values of length parameter are linked to more biais model
    # Lower values of length parameter are linked to flexible (wiggly)

    sigma, length = params[0], params[1]

    # Good method to work with the length-parameters (one l per variable) in the n-dim case
    K = np.zeros((X1.shape[0], X2.shape[0]))
    # for i in range(X1.shape[0]):
    #     K[i, :] = sigma**2 * np.exp(-np.sum((1/(2*length**2)) * (X1[i, :] - X2) ** 2, axis=1))
    # return K
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = sigma**2 * np.exp(-np.sum((1/(2*length**2)) * (x1 - x2) ** 2))
    return K



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

        # self.K : TODO
        # self.L : TODO
        # self.alpha : TODO

        self.theta = None
        # self.theta_low_bds : TODO
        # self.theta_up_bds : TODO

    def negLikelihood(self, theta):
        self.theta = theta
        n = self.X.shape[0]  # Number of training instances

        # Numerically more stable implementation of Eq. (11) as described in
        # http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        y = self.y.ravel()

        # Add noise of the gaussian process sigma_n and nudget noise 1e-10
        K = self.covFunction(self.X, self.X, self.theta)
        K = K + (self.theta[-1]**2 * np.eye(n)) + (1e-10 * np.eye(n))

        # Cholesky factorization and cholesky solving LL.T x = b
        # 1) Ly=b, where y = L.T x
        # 2) L.Tx = y, solve for x
        L, lower = cho_factor(K, lower=True)
        S1 = cho_solve((L, lower), y)
        S2 = cho_solve((L.T, not lower), S1)
        return np.sum(np.log(np.diagonal(L))) + 0.5 * np.dot(y, S2) + 0.5 * n * np.log(2*np.pi)
    

    def gradLogLikelihood(self, theta):
        self.theta = theta
        dlogtheta = np.zeros(3)
        n = self.X.shape[0] 

        # Construct correlation matrix between X_train and X_train
        K = self.covFunction(self.X, self.X, self.theta)
        K = K + (self.theta[-1]**2 * np.eye(n)) + (np.eye(n) * 1e-10)

        # Compute alpha
        L, lower = cho_factor(K, lower=True)
        S1 = cho_solve((L, lower), self.y)
        alpha = cho_solve((L.T, not lower), S1)
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

    def fit(self, X_train, y_train, n_startingPoints=50):

        self.X = X_train 
        self.y = y_train
        
        # Normalize y
        scal_y = StandardNormalization(self.y)
        self.y = scal_y.transform(self.y)
        # Normalize X
        scal_X = MinMaxNormalization(self.X)
        self.X = scal_X.transform(self.X)
 
        # Bounds for the signal variance (sigma_f), length scale (l) and noise variance (sigma_n)
        bounds = [(1e-1, 1e1), (1e-1, 1e1), (1e-4, 1e-1)]
        n_params = 3

        # Generate random starting points (Latin Hypercube)
        lhs = LatinHypercube(n_params).random(n_startingPoints)

        # Scale random samples to the given bounds 
        initial_points = np.zeros(lhs.shape)
        for i in range(n_params):
            lb, ub = bounds[i]
            initial_points[:, i] = (ub-lb) * lhs[:, i] + lb
        # Run local optimizer on all points
        opt_para = np.zeros((n_startingPoints, n_params)) # Record best parameters found for each starting points
        opt_func = np.zeros((n_startingPoints, 1)) # Record best value found for each starting points
        for i in range(n_startingPoints):
            res = minimize(self.negLikelihood, x0=initial_points[i,:], bounds=bounds, jac=self.gradLogLikelihood)
            opt_para[i,:] = res.x
            opt_func[i,:] = res.fun

        # Locate the optimum results and update theta
        self.theta = opt_para[np.argmin(opt_func), :]
        print("Best theta:" + str(self.theta))

        self.X, self.y = scal_X.inv_transform(self.X), scal_y.inv_transform(self.y)

    def updatePosterior(self, xNew, fxNew):
        self.X = np.concatenate((self.X, xNew), axis=0)
        self.y = np.concatenate((self.y, fxNew), axis=0)
        #self.X = np.vstack((self.X, xNew))
        #self.y = np.vstack((self.y, f(xNew)))
        self.fit(self.X, self.y)

    def predict(self, X_test):
        """
        GP model predicting
        """
        # Normalize y with gaussian transformation Z =(z - mean)/sd
        scaler_y = StandardNormalization(self.y)
        self.y = scaler_y.transform(self.y)

        # Normalize X and X_test with min-max transformation
        scaler_X = MinMaxNormalization(self.X)
        self.X = scaler_X.transform(self.X)
        X_test = scaler_X.transform(X_test)

        # Construct correlation matrix betwenn X_train and X_train
        n = self.X.shape[0]
        K = self.covFunction(self.X, self.X, self.theta)
        K = K + (self.theta[-1]**2 * np.eye(n)) + (1e-10 * np.eye(n))

        # Correlation matrix between X_train and X_test
        K_star = self.covFunction(self.X, X_test, self.theta)

        # Correlation matrix between X_test and X_test : snudge noise added to squared matrix
        K_2star = self.covFunction(X_test, X_test, self.theta) + (1e-10 * np.eye(X_test.shape[0]))

        # Warning : cho_solve et cho_factor bug ici, alors on utilise solve_triangular
        L = np.linalg.cholesky(K)
        alpha = solve_triangular(L.T, solve_triangular(L, self.y, lower=True), lower=False)

        # Mean predictions
        mu_pred = np.dot(K_star.T, alpha)

        # Variance predictions
        v = solve_triangular(L, K_star, lower=True)
        variance_pred = K_2star - np.dot(v.T, v)
        variance = np.diag(variance_pred)

        # Inverse transformation
        self.X, self.y = scaler_X.inv_transform(self.X), scaler_y.inv_transform(self.y)
        mu_pred = scaler_y.inv_transform(mu_pred)
        variance = (variance + self.theta[-1]**2) * scaler_y.std  # The sigma_n is added to the predicted variances

        return mu_pred.flatten(), variance

    def score(self, X_test, y_test):
        """Calculate root mean squared error
        Input
        -----
        X_test: test set, array of shape (n_samples, n_features)
        y_test: test labels, array of shape (n_samples, )
        
        Output
        ------
        RMSE: the root mean square error"""
        y_pred, _ = self.predict(X_test)
        RMSE = np.sqrt(np.mean((y_pred - y_test)**2))
        return RMSE


### MAIN ###
if __name__ == '__main__':
    
    #Define the true function that we want to regress on
    # 1D
    # f = lambda x: (x)
    #f = lambda x: (x * np.sin(x))
    #f = lambda x: (x + x**3)
    #f = lambda x: 100*(x**2 * np.exp(-x**2))
    #f = lambda x: (x*6-2)**2*np.sin(x*12-4)
    #f = lambda X, Y: (X**2 + Y**2) * (np.sin(X)- np.cos(Y))
    f = lambda X, Y: (1-X)**2 + 100*(Y- X**2)**2
    
    #domain = np.array([[0, 9]])
    domain = np.array([[-2, 2], [-1, 3]])
    ndim = len(domain)

    n1 = 15 # Number of points to condition on (training points)
    n2 = 20  # Number of points in posterior (test points)
    
    # Training data
    # 1D
    #X_train = np.random.uniform(domain[:, 0], domain[:, 1], size=(n1,ndim))
    #X_train = np.linspace(domain[:, 0], domain[:, 1], n1)
    X_train = (domain[:, 1] - domain[:, 0]) * LatinHypercube(ndim).random(n1) + domain[:, 0]
    #y_train = f(X_train)
    # 2D
    y_train = f(X_train[:, 0], X_train[:, 1])

    # Testing data
    # 1D
    #X_test = np.linspace(domain[:, 0], domain[:, 1], n2)
    #y_test = f(X_test)
    # 2D
    X_test_init = np.linspace(domain[:, 0], domain[:, 1], n2)
    X1, X2 = np.meshgrid(X_test_init[:, 0], X_test_init[:, 1])
    X_test = np.hstack((X1.reshape(-1,1), X2.reshape(-1,1)))
    y_test = f(X_test[:, 0], X_test[:, 1])

    display2D(X_test_init[:, 0], X_test_init[:, 1], y_test.reshape(n2, n2), X_train, y_train, "Objective function")

    #GP model training
    gp = GP()
    gp.fit(X_train, y_train)
    #GP model predicting
    y_pred, variance = gp.predict(X_test)

    display2D(X_test_init[:, 0], X_test_init[:, 1], y_pred.reshape(n2, n2), X_train, y_train, "Prediction")
    #display2Dv2(X_test_init[:, 0], X_test_init[:, 1], y_test.reshape(n2, n2), y_pred.reshape(n2, n2), X_train, y_train)

    #Plot
    #plotGP(X_train, y_train, X_test, y_test, y_pred, variance, figureName="before")

    # #Test add points
    # for i in range(3):
    #     xNew = np.random.uniform(domain[:, 0], domain[:, 1], size=(1, ndim))
    #     # 1D
    #     fxNew = f(xNew)
    #     # 2D
    #     #fxNew = f(xNew[:,0], xNew[:,1])
        
    #     gp.updatePosterior(xNew, fxNew)
    #     y_pred, variance = gp.predict(X_test)
    #     #print("RMSE: " + str(gp.score(X_test, y_test)))
    #     # Plot
    #     #display2D(X_test_init[:, 0], X_test_init[:, 1], y_pred.reshape(n2, n2), X_train, y_train, f"Prediction_{i}")
    #     plotGP(gp.X, gp.y, X_test, y_test, y_pred, variance, figureName="after")

    plt.show()