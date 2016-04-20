import numpy as np 
import seaborn as sea
import scipy 
import matplotlib.pyplot as plt

np.random.seed(1234)

"""
Implementation of Gaussian Process Regression using a toy data set.
There is no optimisation such as gradient descent or MAP.

author: Quan Le, quale@cisco.com
"""

class GP(object):
    """
    The base GP class with functions to calculate the likelihoods.
    """

    def __init__(self, X, y, noise, sigma, l):
        """
        Initalise the GP with the following:

        X: the data in a matrix 
        y: labels
        noise: data variance
        sigma: kernel variance
        l: parmeter for the kernel length

        """

        self.X = X
        self.y = y
        self.noise = noise # sigma_f: var of the data
        self.sigma = sigma # sigma_n: var of the kernel to control the vertical variation 
        self.l = l # kernel length

        # compute the covariance matrix K(x,x)
        self.K = self.kernel(self.X, self.X)

    def kernel(self,a,b, l=None):
        """
        Squared Exponential Kernel where a and b are 1-D vectors 

        l: horizontal length 
        """
        if l == None:
            l = self.l

        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1)-2*np.dot(a,b.T)
        return self.sigma*np.exp(-0.5*sqdist/l**2)

    def log_likelihood(self):
        """
            Compute the negative log likelihood. Parts of the equation requires 
            matrix inversion which is 0(n^2) operation, so we use
            Cholesky decomposition.  
        """

        L = np.linalg.cholesky(self.K + self.noise*np.eye(self.K.shape[0]))
        a_temp = np.linalg.solve(L,y)
        alpha = np.linalg.solve(L.T, a_temp)

        neg_ll = -0.5*( np.dot(y.T, alpha) + \
            np.log(np.linalg.det(self.K + self.noise*np.eye(self.K.shape[0]))) + \
            self.K.shape[0]*np.log(2*np.pi) )

        return L, alpha, neg_ll


    def prediction(self, x_pred, L, alpha):
        """ Calculate the posterior p(f_star | y)"""

        K_star = self.kernel(x, x_pred)
        K_star_star = self.kernel(x_pred, x_pred)

        # calculate the prediction mu
        mu = np.dot(K_star.T, alpha)
        v_temp = np.linalg.solve(L, K_star)
        cov = K_star_star - np.dot(v_temp.T, v_temp)
        return mu, cov

    def sample_prior(self, x_pred, l=None):
        """
        Sample some functions from the prior with the test data from a kernel 
        using x_pred and the length argument.

        This means to generate some covariance in order to generate some random data 
        (the functions sampled) that obeys the properties of the matrix. 
    	"""

        K = self.kernel(x_pred, x_pred, l)                              # compute test kernel K
        L = np.linalg.cholesky(K + 1e-8*np.eye(x_pred.shape[0]))        # use Cholesky instead of matrix inversion
        prior = np.dot(L, np.random.normal(size=(x_pred.shape[0], 10))) # sample the prior 
        
        plt.plot(x_pred, prior)
        plt.title("Samples from the Prior")
        plt.plot()
        

if __name__ == "__main__":
    
    # kernel parameters
    sigma = 2.5
    l = 0.6

    # data noise variance
    sigma_f = 5.9    

    # create the data 
    f = lambda x: x*np.sin(x) + np.random.randn() 
    x = np.linspace(0,8,100).reshape(-1,1)
    y = f(x) # true points
    x_pred = np.random.choice(x.flatten(), size=50)
    x_pred = (np.sort(x_pred)).reshape(-1,1)        # sort and reshape for plotting

    gp = GP(x, y, sigma_f, sigma, l)
    
    # sample the prior
    gp.sample_prior(x_pred, l=1.8)
    L, alpha, neg_ll = gp.log_likelihood()
    mu, cov = gp.prediction(x_pred, L, alpha)
    
    var = np.diag(cov).reshape(-1,1)                # get the variance
    std = np.sqrt(var)                              # std for plotting


    plt.plot(x_pred, mu+(2.*std) , 'g--', label=' $+/- 2 \sigma$ bound')
    plt.plot(x_pred, mu-(2.*std) , 'g--')
    plt.plot(x,y, label='$f(x) = x*sin(x) + \sigma^2$')
    plt.plot(x_pred, mu, 'rx', mew=1, label='predicated points')
    plt.legend(numpoints=1)
    plt.title('Gaussian Process Regression')
    plt.show()

