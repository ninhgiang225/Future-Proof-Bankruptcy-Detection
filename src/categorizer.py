'''classifer.py
Generic categorizing features' type
Ninh Giang Nguyen
CS 251: Data Analysis and Visualization
Spring 2024
'''

import pandas as pd
import numpy as np
from collections import defaultdict

class Categorizer:
    def __init__(self, data = None, thres = 0.8, corr_matrix = None):
        self.data = data
        self.corr_matrix = corr_matrix 
        self.thres = thres
        

    def get_representive_features(self):
        self.corr_matrix = self.data.corr()

        correlated_features = set()

        feature_groups = defaultdict(set)

        # get pairs of highly correlated features
        for i in range(len(self.corr_matrix.columns)):    # get features
            for j in range(i):
                if abs(self.corr_matrix.iloc[i, j]) > self.thres:
                    feature1 = self.corr_matrix.columns[i]
                    feature2 = self.corr_matrix.columns[j]
                    correlated_features.add((feature1, feature2))

        # group highly correlated features into 1 group
        for feature1, feature2 in correlated_features:
            found = False

            for group in feature_groups.values():
                if feature1 in group or feature2 in group:
                    group.update([feature1, feature2])           #add feature 1 and feature 2 in the set
                    found = True
                    break
            
            if not found:
                feature_groups[len(feature_groups)] = {feature1, feature2}

        # get representive feature of each group, the first one
        representive_features =  []

        for group in feature_groups.values():
            representive_features.append(next(iter(group)))
        
        return representive_features
    
    def get_selected_data(self):
        return self.data[self.get_representive_features()]

        


                
                    





        




                       


        
