# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 1 - Main
"""

##standard python libraries
import os
import sys
import warnings
import argparse

##preprocess pipeline
from utils.dataLoaders import LoadCsvData
from utils.preprocess import PreprocessData
from utils.splitData import SplitData
from utils.echoArgs import EchoArgs

##algorithms
from algorithms.naiveClassifier import NaiveClassifier
from algorithms.naiveRegressor import NaiveRegressor

##train/test pipeline
from utils.trainer import TrainModel
from utils.tester import Tester

##metrics
from metrics.metrics import Metrics

##turn off all warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

##command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_name', type = str ,default = '/data',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--dataset_name', type = str ,default = '/abalone',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--namespath', type = str ,default = 'data/abalone.names',
                    help='Path to dataset names'),

parser.add_argument('--discretize_data', type = bool ,default = False,
                    help='Should dataset be discretized?'),

parser.add_argument('--quantization_number', type = int ,default = 3,
                    help='If discretized, then quantization number'),

parser.add_argument('--standardize_data', type = bool , default = True,
                    help='Should data be standardized?'),

parser.add_argument('--k_folds', type = int , default = 5,
                    help='Number of folds for k-fold validation'),

parser.add_argument('--min_examples', type = int , default = 15,
                    help='Drop classes with less examples then this value'),

parser.add_argument('--remove_orig_cat_col', type = bool , default = True,
                    help='Remove the original categorical columns for data encoding'),

parser.add_argument('--modify_train_columns', type = bool , default = False,
                    help='Equal to all columns that are not target, else define train column list'),

parser.add_argument('--train_columns', type = list , default = ['NA'],
                    help='Remove the original categorical columns for data encoding'),

parser.add_argument('--savepath', type = str , default = 'naive_classifier_weights.weights',
                    help='Where to save weights'),

parser.add_argument('--weights_path', type = str , default = 'naive_classifier_weights.weights',
                    help='Where to load weights from')
args = parser.parse_args()


##main body
if __name__ == "__main__":
    ##define paths
    cwd = os.getcwd().replace('\\', '/') ##get current working directory
    data_folder_name = cwd + args.data_folder_name
    datapath = data_folder_name + args.dataset_name + '.data'
    namespath = data_folder_name + args.dataset_name + '.names'
    dataset_name = args.dataset_name
    savepath = args.data_folder_name + '/' +  args.savepath
    weights_path = args.data_folder_name + '/' +  args.weights_path

    ##echo argparse arguments
    EchoArgs(data_folder_name, datapath, namespath, args.dataset_name,
             args.discretize_data,
             args.quantization_number, args.standardize_data, args.k_folds,
             args.min_examples, args.remove_orig_cat_col, args.modify_train_columns,
             args.train_columns).echoJob()

    print('\n******************** ML Pipeline Started ********************')
    ##define tuple of values to drop from dataframe
    values_to_replace = ('na', 'NA', 'nan', 'NaN', 'NAN', '?', ' ')
    values_to_change = {'5more':5, 'more': 5}

    # ##load data
    load_data_obj = LoadCsvData(datapath, namespath, dataset_name)
    names = load_data_obj.loadNamesFromText() ##load names from text
    data = load_data_obj.loadData() ##data to process

    ##preprocess pipeline
    proc_obj = PreprocessData(data, values_to_replace, values_to_change,
                              args.dataset_name, args.discretize_data, args.quantization_number,
                              args.standardize_data, args.remove_orig_cat_col)
    proc_obj.dropRowsBasedOnListValues() ##replaces values from list
    proc_obj.changeValues() ##changes values from values_to_change list
    proc_obj.convertDataType() ##converts datatypes of columns based on what they actually are
    proc_obj.replaceValuesFromListWithColumnMean()##replace value with mean
    df_standard = proc_obj.standardizeData() ##standardizes data
    df_discretized = proc_obj.discretizeData() ##discretizes data
    df_encoded = proc_obj.encodeData() ##encodes data

    ##Split Dataset Pipeline
    split_obj = SplitData(df_encoded, args.k_folds, args.min_examples,
                          args.modify_train_columns, args.train_columns)
    split_obj.removeSparseClasses() ##removes classes that do not meet the min_examples criteria
    split_obj.countDataClasses() ##counts data classes
    split_obj.splitPipeline() ##start of the stratefied k-fold validation split
    train_test_sets = split_obj.createTrainSets() ##k train and test sets returned as a dictionary
    train_columns = split_obj.getTrainColumns() ##gets all columns but target

    ##train and test pipeline
    test_output_dict = {'k_value': [], 'y_test': [], 'predictions': []}
    ##instantiate classifier and regressor and trains/tests each set
    for k in range(len(train_test_sets['train_set'])):
        ##get train data and labels
        X_train = train_test_sets['train_set'][k].loc[:, train_columns]
        y_train = train_test_sets['train_set'][k]['target']

        ##get test data and labels
        X_test = train_test_sets['test_set'][k].loc[:, train_columns]
        y_test = train_test_sets['test_set'][k]['target']

        ##indicate the train test iteration
        print('\nTrain/Test Set: ', k)

        ##instantiate naive classifier model
        classifier = NaiveClassifier(X_train, y_train, len(y_test))

        ##instantiate naive regressor
        regressor = NaiveRegressor(X_train, y_train, len(y_test))

        ##train model
        classifier_train_obj = TrainModel(classifier, X_train, y_train, savepath)
        classifier_trained_model_output = classifier_train_obj.train()

        ##save weights
        classifier_train_obj.saveWeights()

        ##test
        classifier_test_obj = Tester(weights_path, classifier_trained_model_output, X_test, y_test)
        classifier_test_obj.loadWeights() ##loads weights
        classifier_test_output = classifier_test_obj.test() ##gets test output
        ##append test predictions
        test_output_dict['k_value'].append(k)
        test_output_dict['y_test'].append(y_test)
        test_output_dict['predictions'].append(classifier_test_output)

    # ##evaluate model's accuracy classifier
    classifier_metrics_obj = Metrics(test_output_dict, y_train)
    eval_dict = classifier_metrics_obj.evaluate()
    # averaged_by_class = classifier_metrics_obj.averageKfoldByRow()
    metrics_results = classifier_metrics_obj.calculateAccuracy()

    ##evaluate mean-square-error for naive regressor
    regressor_metrics_obj = Metrics(test_output_dict, y_train)
    mse_dict = regressor_metrics_obj.calculateMSE()
    random_mse_dict = regressor_metrics_obj.calculateRandomMSE()



