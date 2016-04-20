import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

from scipy import stats
from collections import Counter

sns.set(color_codes=True)
np.random.seed(1234)

"""
    This is a simple implementation of Infinite Mixture Model or Dirichlet Process
    without optmisation. 
    
    author: Quan Le, quale@cisco.com
"""

class GaussianComponents(object):
    """
    The Gaussian cluster/component class that specifies a cluster's mean.
    """

    def __init__(self, id, mu=None, N=None):

        self.id = id
        self.mu = 0   # cluster means
        self.N = 0    # number of data points in cluster
        if (mu or N) is not None:
            self.mu = mu
            self.N = N

class DirichletProcess(object):
    """
        An Infinite Gaussian Mixture model or Dirichlet Process class.

    """
    def __init__(self, X, alpha, K=2):
        """
        The initalisation function for the DP. It takes the following arguments:

        X:
            a 1-D data vector

        alpha:
            the concentration parameter for the symmetric Dirichlet distribution 
            (emphasises the prior of a data point in a cluster)

        K:
            the number of clusters 
        """

        self.N = X.shape
        self.X = X
        self.alpha = alpha
        self.K = K
        self.cluster_ids = range(K)
        self.var = 0.01

        # randomly assign data points to a cluster
        self.assignments = np.random.randint(0, K, self.N)

        # set up the conjugate priors hyperparams
        self.conj_prior_mu = 0.0
        self.conj_prior_var = 1.0

        # assign each Gaussian object with a cluster ID
        self.components = {cluster_id: GaussianComponents(cluster_id) \
                                    for cluster_id in range(K)} 

        # initalise with even probability for each cluster                           
        self.pi = [self.alpha/K for _ in range(K)]

        self.update_components()


    def update_components(self):
        """ Update the components with a new mean (mu) and N """

        for i in self.components.keys():
            idx = np.where(self.assignments == i)
            mu = self.X[idx].mean()
            N = len(self.X[idx])
            self.components[i].mu = mu
            self.components[i].N = N

    def _create_component(self):
        """ Creates a cluster/component """

        self.K+=1
        cluster_id = max(self.components.keys()) + 1
        new_component = GaussianComponents(cluster_id)

        # add the new component to the dict
        self.components[cluster_id] = new_component
        return cluster_id


    def _remove_component(self):
        """ Remove a cluster """

        for id in self.components.keys():
            if self.components[id].N == 0:
                self.K -=1
                del self.components[id] # delete the component from the dict of components


    def _add_data(self, x, cluster_id):
        """ 
        Add a data point x to a cluster with the given id
        """
        z = self.components[cluster_id]

        #calculate the new mean
        update_mu = (z.mu*z.N + x*1.0) / (z.N+1) 

        #increment number of data points in that cluster
        update_N = z.N + 1                      

        self.components[cluster_id].mu = update_mu
        self.components[cluster_id].N = update_N

    def _remove_data(self, x, cluster_id):
        """
        Remove a data point x from the given cluster
        """

        z = self.components[cluster_id]

        #calculate the new mean 
        update_mu = (z.mu*z.N - x*1.0) / (z.N-1)
        update_N = z.N - 1

        self.components[cluster_id].mu = update_mu
        self.components[cluster_id].N = update_N


    def log_prob_cluster_assignment(self, cluster_id):
        """
        Calculate the log probability of a data point being assigned to the given cluster and
        the current cluster assignment. This is the first term in the equation descirbed in the notebook.
        """
        pi = np.log(self.alpha)
        if cluster_id != -999: 

            # update the cluster with the new probability if an already existing cluster
            pi = np.log(self.components[cluster_id].N)
        return pi

    def log_predictive_likelihood(self, x, cluster_id):
        """
        Calculate the predictive log likelihood that the distribution of the data point x
        belongs to the given cluster. This is the second term.
        """

        # new cluster created for params, never added to list
        if cluster_id == -999: 
            component = GaussianComponents(-999, mu=0, N=0) # -999
        else:
            component = self.components[cluster_id]

        posterior_var = 1.0 / ( (component.N*1.0/self.var) + (1./ self.conj_prior_var) )
        pred_mu = posterior_var * ( (self.conj_prior_mu*1.0 / self.conj_prior_var) + \
                                        (component.N * component.mu * 1.0 / self.var) )
        pred_sigma = np.sqrt(posterior_var + self.var)
        return stats.norm(pred_mu, pred_sigma).logpdf(x)


    def gibbs_sampling(self, n_iter):
        """
        The Collapsed Gibbs sampling to sample data. 
        """

        for i_iter in xrange(n_iter):
            print 'running sampling at iteration:', i_iter
            data_pair = zip(self.X, self.assignments)
            for idx, (x_i, z_i) in enumerate(data_pair):

                # remove the data point
                self._remove_data(x_i, z_i)
                self._remove_component()

                # calculate prob X[i] belonging to each component
                scores = {}
                cluster_ids = self.components.keys() + [-999] # -999 refers to a new cluster made
                for cid in cluster_ids:
                    scores[cid] = self.log_predictive_likelihood(x_i, cid)
                    scores[cid] += self.log_prob_cluster_assignment(cid) # logarithmic so we add for multiplication
                scores = {cid:np.exp(score) for cid, score in scores.iteritems()}
                normalised = 1.0 / sum(scores.values())
                scores = {cid:score * normalised for cid, score in scores.iteritems()}

                # sample new assignment from the scores from marginal distribution
                probs = scores.values()
                labels = scores.keys()
                assign_cluster = np.random.choice(labels, p=probs)
                if assign_cluster == -999:
                    assign_cluster = self._create_component()

                # add the new cluster assignemnt for x_i
                self.assignments[idx] = int(assign_cluster)
                self._add_data(x_i, assign_cluster)

        # return final clustering assignments or labels
        return self.assignments


################################ Utility Function ###########################

def draw_hist(X, labels, it=None):
    """
    Utility function to draw the histogram of the clusters. Can be used to draw 
    all iterations from the Collapsed Gibbs sampling. 
    """

    df = pd.DataFrame({'data':X, 'cluster_assignment':labels}).groupby(by='cluster_assignment')['data']
    hist_data = [df.get_group(cid).tolist() for cid in df.groups.keys()]
    for d in hist_data:
        sns.distplot(d, kde=False)

    if it is None:
        t = 'Original Histogram'
    else:
        t = "Histogram on Iteration " + str(it)
    plt.title(t)
    plt.draw()


def main():
    data_fn = "test_data.pkl"
    data = pickle.load(open(data_fn, "rb"))
    X = data[:,0] * -1.0
    z_label = data[:,1]      # true clustering
    draw_hist(X, z_label)    # plots the true clustering

    n_iter = 50
    model = DirichletProcess(X, 0.1, K=2)
    prob_z = model.gibbs_sampling(50)
    draw_hist(X, prob_z, 50) # plot the final cluster
    plt.show()


if __name__ == "__main__":
    main()




