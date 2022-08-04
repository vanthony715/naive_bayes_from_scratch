#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Naive classifier that output the majority class for any test input
"""
import pandas as pd

class NaiveClassifier:

    '''Test algorthm to predict majority class'''

    def __init__(self, X_train, y_train, test_set_len):
        print('\nNaive Classifier Initialized')
        self.X_train = X_train
        self.y_train = y_train
        self.test_set_len = test_set_len

    def model(self):
        class_cnt_dict = {'class': [], 'count': []}
        classes = list(set(self.y_train))
        for clss in classes:
            cnt = 0
            for sample in self.y_train:
                if sample == clss:
                    cnt+=1
            class_cnt_dict['class'].append(clss)
            class_cnt_dict['count'].append(cnt)

        max_num = max(class_cnt_dict['count'])
        index = class_cnt_dict['count'].index(max_num)
        max_class = class_cnt_dict['class'][index]

        output_list = []
        for i in range(self.test_set_len):
            output_list.append(max_class)

        list_df = pd.DataFrame(output_list)
        return list_df



