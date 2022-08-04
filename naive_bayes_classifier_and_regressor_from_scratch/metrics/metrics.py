#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Computes Metrics
"""
import random
import pandas as pd

class Metrics:
    '''
    Calculates the metrics from test output
    '''
    def __init__(self, test_output_dict, y_train):
        print('\n********************* Results ********************')
        print('\nMetrics Initialized')
        self.test_output_dict = test_output_dict
        self.y_train = y_train
    
    ##this function compares the predictions to the truth values and assigns a 1 for correct or 0 for incorrect
    ##this is only used for classification
    def evaluate(self):
        print('Evaluating...')
        self.test_output_dict['result'] = []
        for k in self.test_output_dict['k_value']:
            results_list = [] ##record keeping for right/wrong predictions
            for i in range(len(self.test_output_dict['predictions'][k][0])):
                if self.test_output_dict['y_test'][k][i] == self.test_output_dict['predictions'][k][0][i]:
                    results_list.append(1)
                else:
                    results_list.append(0)
            self.test_output_dict['result'].append(results_list)
        return self.test_output_dict

    def getRawMetrics(self):
        ##place holder for tp, fp, fn ... 
        pass

    def calculatePrecision(self):
        ##place holder for calculating precision
        pass

    def calculateRecall(self):
        ##place holder for calculating recall
        pass

    def calculateF1(self):
        ##place holder for calculating F1 score
        pass
    
    ##This function calculate the accuracy of the model's output of the test set for classification
    def calculateAccuracy(self):
        self.k_folds = list(set(self.test_output_dict['k_value']))
        sample_cnt = len(self.test_output_dict['result'][0])
        for k in self.k_folds:
            correct_cnt = 0
            for i in range(len(self.test_output_dict['result'][k])):
                if self.test_output_dict['result'][k][i] == 1:
                    correct_cnt += 1

            accuracy = round((correct_cnt / sample_cnt) * 100, 2)
            print('Test Set: ', k, 'Test Set Accuracy (%): ', accuracy)
        return self.test_output_dict
    
    ##This function calculates mse and is only used on regression
    def calculateMSE(self):
        print('Evaluating...')
        ##keep track of k value and mse
        mse_dict = {'k_value': [], 'mse': []}

        ##iterate through test sets
        for k in self.test_output_dict['k_value']:
            error_list = [] ##record keeping for y - y_hat calculations
            for i in range(len(self.test_output_dict['predictions'][k][0])):
                ##find y - y_hat for each y in test set
                error = self.test_output_dict['y_test'][k][i] - self.test_output_dict['predictions'][k][0][i]
                error_sqrd = error**2  ##error squared
                error_list.append(error_sqrd) ##record

            ##mse = 1/n(sum(y - y_hat)^2)
            mse = sum(error_list) / len(self.test_output_dict['predictions'][k][0])
            print('Test Set: ', k, 'MSE: ', mse)
            ##record values
            mse_dict['k_value'].append(k)
            mse_dict['mse'].append(mse)

        ##take the average of each mse value per test set
        avg_mse = sum(mse_dict['mse']) / len(self.test_output_dict['k_value'])
        print('Averaged MSE: ', avg_mse)
        print('\n')
        return mse_dict
    
    ##This function randomly generates a target value scene in the train set and uses that value for the prediction value
    ##this is only used for regression
    def calculateRandomMSE(self):
        print('Evaluating...')
        ##keep track of k value and mse
        mse_dict = {'k_value': [], 'mse': []}

        ##iterate through test sets
        for k in self.test_output_dict['k_value']:
            error_list = [] ##record keeping for y - y_hat calculations
            for i in range(len(self.test_output_dict['predictions'][k][0])):
                ##get random sample of possible y values in test set
                random_prediction = random.sample(list(self.y_train), 1)
                ##find y - y_hat for each y in test set
                error = self.test_output_dict['y_test'][k][i] - random_prediction
                error_sqrd = error**2  ##error squared
                error_list.append(error_sqrd) ##record

            ##mse = 1/n(sum(y - y_hat)^2)
            mse = sum(error_list) / len(self.test_output_dict['predictions'][k][0])
            print('Test Set: ', k, 'Random Value MSE: ', mse)
            ##record values
            mse_dict['k_value'].append(k)
            mse_dict['mse'].append(mse)

        ##take the average of each mse value per test set
        avg_mse = sum(mse_dict['mse']) / len(self.test_output_dict['k_value'])
        print('Averaged Random Value MSE: ', avg_mse)
        return mse_dict






