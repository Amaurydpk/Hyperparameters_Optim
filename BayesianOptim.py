import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from sklearn.feature_selection import SelectFdr
from GaussianProcess import GP




def expectedImprovement(X, gp, currentBestEval):
    """
    Computes the EI at points X using a Gaussian process surrogate model.
    """
    mu, _, std = gp.predict(X)
    delta = mu - currentBestEval
    # x / inf = 0
    std[std == 0] = np.inf
    z = delta / std
    return delta * norm.cdf(z) + std * norm.pdf(z)


def new_proposal(acquisition, X_sample, y_sample, gpr, bounds, n=25):
    """
    
    """

    def f(x):
        return expectedImprovement(x[None, :], gpr, self.y.min())

    x0 = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n, self.X.shape[1]))
    proposal = None
    best_ei = np.inf
    for x0_ in x0:
        res = minimize(f, x0_, bounds=self.bounds)
        if res.success and res.fun < best_ei:
            best_ei = res.fun
            proposal = res.x
        if np.isnan(res.fun):
            raise ValueError("NaN within bounds")
    return proposal
    


   

### MAIN ###
if __name__ == '__main__':
    
    # Define the true function that we want to regress on
    #f = lambda x: (x)        
    #f = lambda x: (x * np.sin(x))    
    #f = lambda x: (x + x**3)
    #f = lambda x: 100*(x**2 * np.exp(-x**2))
    f = lambda x: (x*6-2)**2*np.sin(x*12-4)

    domain = np.array([0, 1])
    n1 = 2 # Number of points to condition on (training points)
    n2 = 100  # Number of points in posterior (test points)
    nTrials = 8 # Number of run for BO
    
    # Training data
    # X_train = np.random.uniform(domain[0], domain[1], size=(n1,1))
    X_train = np.linspace(domain[0], domain[1], n1).reshape(-1,1)
    y_train = f(X_train)

    # Testing data
    X_test = np.linspace(domain[0], domain[1], n2).reshape(-1,1)
    y_test = f(X_test)

    for i in range(2, 10):
        print(f"Iteration {i}")
   











   


    
