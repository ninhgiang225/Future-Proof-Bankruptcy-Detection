'''em.py
Cluster data using the Expectation-Maximization (EM) algorithm with Gaussians
Ninh Giang Nguyen
CS 252: Mathematical Data Analysis Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from IPython.display import display, clear_output
from mpl_toolkits.mplot3d import Axes3D



class EM():
    def __init__(self, data=None):
        '''EM object constructor.
        See docstrings of individual methods for what these variables mean / their shapes

        (Should not require any changes)
        '''
        self.k = None
        self.centroids = None
        self.cov_mats = None
        self.responsibilities = None
        self.pi = None
        self.data_centroid_labels = None

        self.loglikelihood_hist = None
        self.out_liers = []
        self.data = data
        self.num_samps = None
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape
        
        

    def gaussian(self, pts, mean, sigma):
        '''
        Evaluates a multivariate Gaussian distribution described by
        mean `mean` and covariance matrix `sigma` at the (x, y) points `pts`

        Parameters:
        -----------
        pts: ndarray. shape=(num_samps, num_features).
            Data samples at which we want to evaluate the Gaussian
            Example for 2D: shape=(num_samps, 2)
        mean: ndarray. shape=(num_features,)
            Mean of Gaussian (i.e. mean of one cluster). Same dimensionality as data
            Example for 2D: shape=(2,) for (x, y)
        sigma: ndarray. shape=(num_features, num_features)
            Covariance matrix of a Gaussian (i.e. covariance of one cluster).
            Example for 2D: shape=(2,2). For standard deviations (sigma_x, sigma_y) and constant c,
                Covariance matrix: [[sigma_x**2, c*sigma_x*sigma_y],
                                    [c*sigma_x*sigma_y, sigma_y**2]]

        Returns:
        -----------
        ndarray. shape=(num_samps,)
            Multivariate gaussian evaluated at the data samples `pts`
        '''
        if pts.ndim == 1:
            m = len(mean)
        else:
            m = pts.shape[1]
        n = len(pts)
        a = (2*np.pi)**(m/2) *  (np.linalg.det(sigma))**(1/2)

        # plug just 1 data sample xi
        if len(pts.shape) == 1: 
            b = -1/2 * np.sum((pts - mean) @ np.linalg.inv(sigma) * (pts - mean)) 
        else:  # plug dataset A
            b = np.zeros(n)
            for i in range(n):
                b[i] = -1/2 * (pts[i] - mean) @ np.linalg.inv(sigma) @ (pts[i] - mean).T

        return np.exp(b) / a

    def initalize(self, k):
        '''Initialize all variables used in the EM algorithm.

        Parameters:
        -----------
        k: int. Number of clusters.

        Returns
        -----------
        None

        TODO:
        - Set k as an instance variable.
        - Initialize the log likelihood history to an empty Python list.
        - Initialize the centroids to random data samples
            shape=(k, num_features)
        - Initialize the covariance matrices to the identity matrix
        (1s along main diagonal, 0s elsewhere)
            shape=(k, num_features, num_features)
        - Initialize the responsibilities to an ndarray of 1/k.
            shape=(k, num_samps)
        - Initialize the pi array (proportion of points assigned to each cluster) so that each cluster
        is equally likely.
            shape=(k,)
        '''
        self.k = k
        self.loglikelihood_hist = []

        random_centroids = np.random.choice(self.num_samps, k)
        self.centroids = self.data[random_centroids]

        self.cov_mats = np.zeros([self.k , self.num_features, self.num_features])
        self.pi = np.zeros([self.k, 1])
        self.responsibilities = np.zeros([self.k, self.num_samps])/self.k

        for i in range (self.k):
            self.pi[i] = 1/self.k
            for j in range(self.num_features):
                self.cov_mats[i][j][j] = 1

    def e_step(self, method="Gaussian"):
        '''Expectation (E) step in the EM algorithm.
        Set self.responsibilities, the probability that each data point belongs to each of the k clusters.
        i.e. leverages the Gaussian distribution.

        NOTE: Make sure that you normalize so that the probability that each data sample belongs
        to any cluster equals 1.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.responsibilities: ndarray. shape=(k, num_samps)
            The probability that each data point belongs to each of the k clusters.
        '''
        for i in range (self.k):
            if method == "Gaussian":
                self.responsibilities[i] = self.gaussian(self.data, self.centroids[i], self.cov_mats[i])
            if method == "Exponential":
                self.responsibilities[i] = self.exponential(self.data, self.centroids[i])
        
        for j in range (self.num_samps):
            self.responsibilities[:, j] /= np.sum(self.responsibilities[:, j])

    
        return self.responsibilities


    def m_step(self):
        '''Maximization (M) step in the EM algorithm.
        Set self.centroids, self.cov_mats, and self.pi, the parameters that define each Gaussian
        cluster center and spread, as well as the degree to which data points "belong" to each cluster

        TODO:
        - Compute the proportion of data points that belong to each cluster.
        - Compute the mean of each cluster. This is the mean over all data points, but weighting
        the data by the probability that they belong to that cluster.
        - Compute the covariance matrix of each cluster. Use the usual equation (for all the data),
        but before summing across data samples, make sure to weight each data samples by the
        probability that they belong to that cluster.

        NOTE: When computing the covariance matrix, use the updated cluster centroids for
        the CURRENT time step.

        Parameters:
        -----------
        None

        Returns
        -----------
        self.centroids: ndarray. shape=(k, num_features)
            Mean of each of the k Gaussian clusters
        self.cov_mats: ndarray. shape=(k, num_features, num_features)
            Covariance matrix of each of the k Gaussian clusters
            Example of a covariance matrix for a single cluster (2D data): [[1, 0.2], [0.2, 1]]
        self.pi: ndarray. shape=(k,)
            Proportion of data points belonging to each cluster.
        '''
        # Update pi
        self.pi = np.mean(self.responsibilities, axis=1)

        # Update centroids
        self.centroids = np.dot(self.responsibilities, self.data) / np.sum(self.responsibilities, axis=1, keepdims=True)

        # Update covariance matrices
        for i in range(self.k):
            diff = self.data - self.centroids[i, np.newaxis, :]
            weighted_diff = self.responsibilities[i, :, np.newaxis] * diff
            self.cov_mats[i] = np.dot(weighted_diff.T, diff) / np.sum(self.responsibilities[i])

        return self.centroids, self.cov_mats, self.pi


    def log_likelihood(self):
        '''Compute the sum of the log of the Gaussian probability of each data sample in each cluster
        Used to determine whether the EM algorithm is converging.

        Parameters:
        -----------
        None

        Returns
        -----------
        float. Summed log-likelihood of all data samples

        NOTE: Remember to weight each cluster's Gaussian probabilities by the proportion of data
        samples that belong to each cluster (pi).
        '''
        
        log_likelihood_sum = 0.0
        for j in range(self.k):
            weighted_gaussian_prob = self.pi[j]* self.gaussian(self.data, self.centroids[j], self.cov_mats[j])
            log_likelihood_sum += weighted_gaussian_prob
            
        
        log_likelihood_sum = np.sum(np.log(log_likelihood_sum))
     


        return log_likelihood_sum
    

    def cluster(self, k=2, max_iter=100, stop_tol=1e-3, verbose=False, animate=False, method = "Gaussian"):
        '''Main method used to cluster data using the EM algorithm
        Perform E and M steps until the change in the loglikelihood from last step to the current
        step <= `stop_tol` OR we reach the maximum number of allowed iterations (`max_iter`).

        Parameters:
        -----------
        k: int. Number of clusters.
        max_iter: int. Max number of iterations to allow the EM algorithm to run.
        stop_tol: float. Stop running the EM algorithm if the change of the loglikelihood from the
        previous to current step <= `stop_tol`.
        verbose: boolean. If true, print out the current iteration, current log likelihood,
            and any other helpful information useful for debugging.

        Returns:
        -----------
        self.loglikelihood_hist: Python list. The log likelihood at each iteration of the EM algorithm.

        NOTE: Reminder to initialize all the variables before running the EM algorithm main loop.
            (Use the method that you wrote to do this)
        NOTE: At the end, print out the total number of iterations that the EM algorithm was run for.
        NOTE: The log likelihood is a NEGATIVE float, and should increase (approach 0) if things are
            working well.
        '''
        self.initalize(k)
        cur_iter = 0
        
        while cur_iter <= max_iter:
            cur_iter += 1
            prev_log = self.log_likelihood()
            self.e_step(method)
            self.m_step()
            cur_log = self.log_likelihood()
            self.loglikelihood_hist.append(cur_log)

            if animate:
                clear_output(wait=True)
                plt.pause(0.1)
                self.plot_clusters(self.data)

            if np.abs(cur_log - prev_log) <= stop_tol:
                break
        if verbose: 
            print("The iteration is ", cur_iter)
            print("The log likelyhood is ", cur_log)

        return self.loglikelihood_hist


    def find_outliers(self, thres=0.05):
        '''Find outliers in a dataset using clustering by EM algorithm

        Parameters:
        -----------
        thres: float. Value >= 0
            Outlier defined as data samples assigned to a cluster with probability of belonging to
            that cluster < thres

        Returns:
        -----------
        Python lists of ndarrays. len(Python list) = len(cluster_inds).
            Example if k = 2: [(array([ 0, 17]),), (array([20, 26]),)]
                The Python list has 2 entries. Each entry is a ndarray.
            Within each ndarray, indices of `self.data` of detected outliers according to that cluster.
                For above example: data samples with indices 20 and 26 are outliers according to
                cluster 2.
        '''

        outlier_indices = []
        indices = np.arange(self.num_samps)
        labels = np.argmax(self.responsibilities, axis=0)

        for c in range(self.k):
            cur_probs = self.gaussian(self.data[labels == c], self.centroids[c], self.cov_mats[c])
            cur_indices =  indices[labels == c]
            outlier_indices.append(cur_indices[cur_probs <= thres])

        return outlier_indices




##############################################################################



    def estimate_log_probs(self, xy_points):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = np.log(self.gaussian(xy_points, self.centroids[c], self.cov_mats[c]))
        probs += np.log(self.pi[:, np.newaxis])
        return -logsumexp(probs, axis=0)

    def get_sample_points(self, data, res):
        '''Used for plotting the clusters.

        (Should not require any changes)
        '''
        data_min = np.min(data, axis=0) - 0.5
        data_max = np.max(data, axis=0) + 0.5
        x_samps, y_samps = np.meshgrid(np.linspace(data_min[0], data_max[0], res),
                                       np.linspace(data_min[1], data_max[1], res))
        plt_samps_xy = np.c_[x_samps.ravel(), y_samps.ravel()]
        return plt_samps_xy, x_samps, y_samps
    


    def plot_clusters(self, data, res=100, show=True):
        '''Method to call to plot the clustering of `data` using the EM algorithm

        (Should not require any changes)
        '''
        # Plot points assigned to each cluster in a different color
        cluster_hard_assignment = np.argmax(self.responsibilities, axis=0)
        for c in range(self.k):
            curr_clust = data[cluster_hard_assignment == c]
            plt.plot(curr_clust[:, 0], curr_clust[:, 1], '.', markersize=7)

        # Plot centroids of each cluster
        plt.plot(self.centroids[:, 0], self.centroids[:, 1], '+k', markersize=12)

        # Get grid of (x,y) points to sample the Gaussian clusters
        xy_points, x_samps, y_samps = self.get_sample_points(data, res=res)
        
        # Evaluate the sample points at each cluster Gaussian. For visualization, take max prob
        # value of the clusters at each point
        probs = np.zeros([self.k, len(xy_points)])
        for c in range(self.k):
            probs[c] = self.gaussian(xy_points, self.centroids[c], self.cov_mats[c])
        probs /= probs.max(axis=1, keepdims=True)
        probs = probs.sum(axis=0)
        probs = np.reshape(probs, [res, res])

        # Make heatmap for cluster probabilities
        plt.contourf(x_samps, y_samps, probs, cmap='viridis')
        if show:
            plt.show()


    def exponential(self, pts, mean):
        if len(pts.shape) == 1: 
            return 1/mean * np.exp(- pts/mean)
        else:  # plug dataset A
            return (1/mean * np.exp(- pts/mean))[:, -1]
        

############################


    def plot_contour(self, data_vars, sig_level, fig_sz=(12, 12), adjust=0.1, title=''):
        plt.figure(figsize=fig_sz)
        
        k = 1
        for i in range(len(data_vars)):
            for j in range(i):
                
                the_data = self.data[[data_vars[i], data_vars[j]]].to_numpy()

                self.k = 1

                mean = np.mean(the_data, axis = 0)
                plt_samps_xy, x_samps, y_samps = self.get_sample_points(the_data, res = 100)
                prob = self.gaussian(plt_samps_xy, mean, np.cov(the_data, rowvar=False))
           
                prob = prob.reshape(x_samps.shape)


                plt.subplot(1, len(data_vars), k)
                plt.plot(the_data[:, 0], the_data[:, 1], 'x', markersize=0.1, c="white")
                plt.plot(mean[0], mean[1], '+k', markersize=13)
                plt.contourf(x_samps, y_samps, prob, cmap='viridis')

                q1 = round(np.quantile(the_data[0], sig_level), 3) - adjust
                q2 = round(np.quantile(the_data[0], 1-sig_level), 3) + adjust

                p1 = round(np.quantile(the_data[1], sig_level), 3) - adjust
                p2 = round(np.quantile(the_data[1],  1-sig_level), 3) + adjust

                plt.xlim(q1, q2)
                plt.ylim(p1, p2)

                plt.xlabel(data_vars[i], size = 10)
                plt.ylabel(data_vars[j], size = 10)

                plt.tight_layout()

                self.plot_outliers(the_data)

                
                k += 1
                if k > len(data_vars):
                    return
                

                

    def plot_outliers(self, data, thres = 0.05):
        probs = self.gaussian(data, np.mean(data, axis = 0), np.cov(data, rowvar=False))

        for i, prob in enumerate(probs):
            if prob  <= thres:
                self.out_liers.append(i)
                plt.scatter(data[i,0], data[i,1], s = 4, color = "red", marker = "+")
    
    def get_outliers_index(self):
        return self.out_liers

        

        
        
  


