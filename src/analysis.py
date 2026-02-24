'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Ninh Giang Nguyen
CS 251/2: Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})
        

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        pass

        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        if len(rows) == 0:
            rows = range(self.data.get_num_samples())
        data_subset = self.data.select_data(headers, rows)
        return np.min(data_subset, axis = 0)


    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        if len(rows) == 0:
            rows = range(self.data.get_num_samples())
        data_subset = self.data.select_data(headers, rows)
        return np.max(data_subset, axis = 0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        mins = self.min(headers, rows)
        maxes = self.max(headers, rows)
        return mins, maxes
    
    def median(self, headers, rows=[]):
        if len(rows) == 0:
            rows = range(self.data.get_num_samples())
        data_subset = self.data.select_data(headers, rows)
        return np.median(data_subset, axis=0)
    


    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''

        if len(rows) == 0:
            rows = range(self.data.get_num_samples())
        data_subset = self.data.select_data(headers, rows)
        return np.sum(data_subset, axis=0) / len(rows)
    

    def weighted_mean(self, headers, weights, rows=[]):
        if len(rows) == 0:
            rows = range(self.data.get_num_samples())
        data_subset = self.data.select_data(headers, rows)

        weighted_values = data_subset * np.array(weights)[:, np.newaxis]
        sum_weighted_values = np.sum(weighted_values, axis=0)
        sum_weights = np.sum(weights)

        return sum_weighted_values / sum_weights


    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE:
        - You CANNOT use np.var or np.mean here!
        - There should be no loops in this method!
        '''

        if len(rows) == 0:
            rows = range(self.data.get_num_samples())
        data_subset = self.data.select_data(headers, rows)
        means = self.mean(headers, rows)

        return np.sum(np.square(data_subset - means), axis=0) / (len(rows) -1)
        

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over, or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE:
        - You CANNOT use np.var, np.std, or np.mean here!
        - There should be no loops in this method!
        '''
        return np.sqrt(self.var(headers, rows))


    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var,size, title=""):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and "y" variable in the dataset
        `dep_var`. Both `ind_var` and `dep_var` should be strings in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        ind_index = self.data.get_mappings()[ind_var]
        dep_index = self.data.get_mappings()[dep_var]
        x = self.data.select_data_2([ind_index])
        y = self.data.select_data_2([dep_index])
        plt.scatter(x, y,s=size)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)

        return x, y


    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in `data_vars` in the
        x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        1. Make the len(data_vars) x len(data_vars) grid of scatterplots
        2. The y axis of the FIRST column should be labeled with the appropriate variable being plotted there.
        The x axis of the LAST row should be labeled with the appropriate variable being plotted there.
        3. Only label the axes and ticks on the FIRST column and LAST row. There should be no labels on other plots
        (it looks too cluttered otherwise!).
        4. Do have tick MARKS on all plots (just not the labels).
        5. Because variables may have different ranges, your pair plot should share the y axis within columns and
        share the x axis within rows. To implement this, add
            sharex='col', sharey='row'
        to your plt.subplots call.

        NOTE: For loops are allowed here!
        '''
   

        num_vars = len(data_vars)
        fig, axes = plt.subplots(nrows=num_vars, ncols=num_vars, figsize=fig_sz, sharex='col', sharey='row')
        

        for i, var_i in enumerate(data_vars):
            for j, var_j in enumerate(data_vars):
                var_x = []
                var_y = []
                var_x.append(var_i)
                var_y.append(var_j)
                # axes[i][j].set_xlim(-5, 5)
                # axes[i][j].set_ylim(-5, 5)
    

                axes[i,j].scatter(x = self.data[var_y], y = self.data[var_x], marker = "x", c= "green")
                if j == 0:
                    axes[i, j].set_ylabel(var_i, size = 10)
                if i == num_vars - 1:
                    axes[i, j].set_xlabel(var_j, size = 10)


        plt.subplots_adjust(wspace=0.5)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        return fig, axes
    
    
    def range(self, array,  sig_level = 0.05):
        'input: dataFrame'

        if isinstance(array, (pd.Series, pd.DataFrame)):
            q1 = round(array.quantile(sig_level).mean(), 3)
            q3 = round(array.quantile(1 - sig_level).mean(), 3)
        
        elif isinstance(array, np.ndarray):
            q1 = round(np.quantile(array, sig_level), 3)
            q3 = round(np.quantile(array, 1 - sig_level), 3)
        
        else:
            print("array's neither numpy or pandas array.")

        return  (q1, q3), round(q3-q1, 3)
        



        
