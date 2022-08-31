import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from xxhash import xxh64_intdigest
from GaussianProcess import GP, plotGP



def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X)
    plt.fill_between(X.flatten(), 
                    mu - 1.96 * np.sqrt(std), 
                    mu + 1.96 * np.sqrt(std),
                    facecolor='lightgrey',
                    label='95% Credibility Interval')
    plt.tick_params(axis='both', which='major', labelsize=12) 
    plt.plot(X, Y, 'r--', linewidth=1, label='Test Function')
    plt.plot(X, mu, 'b-', linewidth=1, label='GP Prediction')
    plt.plot(X_sample, Y_sample, 'ro', markerfacecolor='r', label='Samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend(loc="best")

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'g-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend(loc="best")  


def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 5))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)
    
    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_min_watermark = np.minimum.accumulate(y)
    
    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_min_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')


def expectedImprovement(X, gp, currentBestEval, xi=0.01):
    """
    Computes the EI at points X using a Gaussian process surrogate model.
    """
    mu, std = gp.predict(X)
    #delta = mu - currentBestEval # Maximize
    delta = currentBestEval - mu # Minimize
    z = delta / std
    ei = delta * norm.cdf(z) + std * norm.pdf(z)
    ei[std == 0.0] = 0.0
    return ei


def new_proposal(acquisition, gp, bounds, n_restart=25):

    def min_obj(x):
        return -acquisition(x, gp, np.min(gp.y))

    proposal = None
    best_obj = np.inf
    for x0 in np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_restart, gp.X.shape[1])):
        res = minimize(min_obj, x0, bounds=bounds, method='L-BFGS-B')
        if res.success and res.fun < best_obj:
            best_obj = res.fun
            proposal = res.x
        if np.isnan(res.fun):
            raise ValueError("NaN within bounds")
    return proposal


### MAIN ###
if __name__ == '__main__':

    # Define the true function that we want to regress on
    # f = lambda x: (x)
    # fNumber = 1        
    f = lambda x: (x * np.sin(x))
    # fNumber = 2     
    # f = lambda x: (x + x**3)
    # fNumber = 3
    #f = lambda x: -100*(x**2 * np.exp(-x**2))
    #fNumber = 4
    #f = lambda x: 1/2*(x*6-2)**2*np.sin(x*12-4)
    # fNumber = 5
    #f = lambda X, Y: (1-X)**2 + 100*(Y- X**2)**2
    fNumber = 6
    
    domain = np.array([[2, 8]])
    #domain = np.array([[-10, 2], [-1, 3]])
    
    #domain = np.array([[0, 10], [0, 10]])
    ndim = len(domain)
    n1 = 3 # Number of points to condition on (training points)
    n2 = 100  # Number of points in posterior (test points)
    
    # Training data
    #X_init = np.random.uniform(domain[:, 0], domain[:, 1], size=(n1,ndim))
    X_init = (domain[:, 1] - domain[:, 0]) * LatinHypercube(ndim).random(n1) + domain[:, 0]
    #X_init = np.linspace(domain[:, 0], domain[:, 1], n1)
    #y_init = f(X_init[:, 0], X_init[:, 1])
    y_init = f(X_init)
    
    # Testing data
    X_test = np.linspace(domain[:, 0], domain[:, 1], n2)
    y_test = f(X_test)
    #y_test = f(X_test[:, 0], X_test[:, 1])

    #GP model training
    gp = GP()
    
    X_sample, y_sample = X_init, y_init 

    nIter = 6

    plt.figure(figsize=(12, nIter * 3))
    plt.subplots_adjust(hspace=0.4) 

    for i in range(nIter):
        print(f'Iteration {i+1}')
        # Update Gaussian process with existing samples
        gp.fit(X_sample, y_sample)
        # Obtain next sampling point from the acquisition function (expected_improvement)
        xNext = new_proposal(expectedImprovement, gp, domain)
        # Obtain next sample value from the objective function
        yNext = f(xNext) # black box eval
        print(xNext, yNext)

        #Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(nIter, 2, 2 * i + 1)
        plot_approximation(gp, X_test, y_test, X_sample, y_sample, xNext, show_legend=i==0)
        plt.title(f'Iteration {i+1}')

        plt.subplot(nIter, 2, 2 * i + 2)
        plot_acquisition(X_test, expectedImprovement(X_test, gp, np.min(y_sample)), xNext, show_legend=i==0)
    
        # Add sample to previous samples
        X_sample = np.vstack((X_sample, xNext))
        y_sample = np.vstack((y_sample, yNext))
        #plt.savefig('./images/' + f"Fun{fNumber}_Run_Init{n1}_Iter{nIter}")
        plt.savefig('./images/' + "Runs")
    
    plot_convergence(X_sample, y_sample, n_init=n1)
    #plt.savefig('./images/' + f"Fun{fNumber}_Conv_Init{n1}_Iter{nIter}")
    plt.savefig('./images/' + "Conv")
    