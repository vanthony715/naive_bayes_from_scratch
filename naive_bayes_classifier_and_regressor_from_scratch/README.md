run: python vasquez_project_1_main.py [--arguments]

Output will print in the terminal to indicate job information, process step, and evaluation metrics.

arguments include:

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