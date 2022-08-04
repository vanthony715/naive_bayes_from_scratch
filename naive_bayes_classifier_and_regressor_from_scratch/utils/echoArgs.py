#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Machine Learning

Description: Echos argparse arguments
"""

class EchoArgs:
    
    '''
    Echos Arguments
    '''
    def __init__(self, data_folder_name, datapath, namespath, dataset_name, discretize_data,
                 quantization_number, standardize_data, k_folds, min_examples,
                 remove_orig_cat_col, modify_train_columns, train_columns):
        self.data_folder_name = data_folder_name
        self.datapath = datapath
        self.namespath = namespath
        self.dataset_name = dataset_name
        self.discretize_data = discretize_data
        self.quantization_number = quantization_number
        self.standardize_data = standardize_data
        self.k_folds = k_folds
        self.min_examples = min_examples
        self.remove_orig_cat_col = remove_orig_cat_col
        self.modify_train_columns = modify_train_columns
        self.train_columns = train_columns
        
    def echoJob(self):
        print('\n--------------------- Job Description --------------------')
        print('Data folder name: ', self.data_folder_name)
        print('Path to data: ', self.datapath)
        print('Dataset name: ', self.dataset_name)
        print('Names file name: ', self.namespath)
        print('Discretize data?: ', self.discretize_data)
        print('Quantization number: ', self.quantization_number)
        print('Standardize data: ', self.standardize_data)
        print('K-Folds: ', self.k_folds)
        print('Min number of examples: ', self.min_examples)
        print('Remove original cat column when decoding?: ', self.remove_orig_cat_col)
        print('Modify the train columns?: ', self.modify_train_columns)
        print('Data folder name: ', self.data_folder_name)
        print('Selected train columns: ', self.train_columns)
        print('----------------------------------------------------------')