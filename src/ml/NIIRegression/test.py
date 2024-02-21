# NIIRegressor test module

print('initializing...')  # noqa

# Code destined to testing NII
# regression neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from keras.metrics import Recall
from keras.models import load_model
from keras.metrics import Precision
from argparse import ArgumentParser
from keras.metrics import BinaryAccuracy
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import normalize_data
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_data_split_regression
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'NIIRegressor test module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # dataset file param
    parser.add_argument('-d', '--dataset-file',
                        dest='dataset_file',
                        required=True,
                        help='defines path to dataset df (.csv) file')

    # predictions file param
    parser.add_argument('-p', '--predictions-file',
                        dest='predictions_file',
                        required=True,
                        help='defines path to prediction df (.csv) file')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_metrics_df(test_df: DataFrame,
                   predictions_df: DataFrame
                   ) -> DataFrame:
    pass


def print_metrics(df: DataFrame) -> None:
    pass


def nii_regression_test(dataset_file: str,
                        predictions_file: int
                        ) -> None:
    """
    Given a path to dataset df, and
    a path to a file containing test
    data predictions, prints metrics
    on console.
    """
    # reading dataset df
    print('reading dataset df...')
    dataset_df = read_csv(dataset_file)

    # filtering df by test data
    print('filtering df by test data...')
    filtered_df = dataset_df[dataset_df['split'] == 'test']

    # reading predictions df
    print('reading predictions df...')
    predictions_df = read_csv(dataset_file)

    # getting metrics df
    metrics_df = get_metrics_df(test_df=filtered_df,
                                predictions_df=predictions_df)

    # printing metrics on console
    print_metrics(df=metrics_df)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting dataset file param
    dataset_file = args_dict['dataset_file']

    # predictions file param
    predictions_file = args_dict['predictions_df']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    using_gpu = is_using_gpu()
    using_gpu_str = f'Using GPU: {using_gpu}'
    print(using_gpu_str)

    # waiting for user input
    enter_to_continue()

    # running nii_regression_test function
    nii_regression_test(dataset_file=dataset_file,
                        predictions_file=predictions_file)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
