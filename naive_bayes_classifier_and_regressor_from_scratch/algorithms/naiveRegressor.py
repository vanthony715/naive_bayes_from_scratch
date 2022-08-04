#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Naive regressor that outputs the mean value of the trainset for any test input
"""
import pandas as pd

class NaiveRegressor:

    '''Test algorthm to predict average'''

    def __init__(self, X_train, y_train, test_set_len):
        print('Naive Regressor Initialized')
        self.X_train = X_train
        self.y_train = y_train
        self.test_set_len = test_set_len

    def model(self):
        mean = self.y_train.mean()

        output_list = []
        for i in range(self.test_set_len):
            output_list.append(mean)

        list_df = pd.DataFrame(output_list)
        return list_df
