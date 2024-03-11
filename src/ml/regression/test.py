# regression test module

print('initializing...')  # noqa

# Code destined to testing
# regression neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from math import sqrt
from pandas import merge
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import enter_to_continue
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
    description = 'regression test module'

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


def get_test_df(dataset_file: str) -> DataFrame:
    """
    Given a path to a dataset file,
    returns filtered df containing
    only test data and required cols
    for analysis.
    """
    # reading dataset df
    dataset_df = read_csv(dataset_file)

    # filtering df by test data
    filtered_df = dataset_df[dataset_df['split'] == 'test']

    # defining cols to keep
    cols_to_keep = ['crop_name',
                    'class']

    # dropping unrequired cols
    filtered_df = filtered_df[cols_to_keep]

    # returning filtered df
    return filtered_df


def get_predictions_df(predictions_file: str) -> DataFrame:
    """
    Given a path to a predictions file,
    returns loaded df filtered by cols
    related to test analysis.
    """
    # reading predictions df
    predictions_df = read_csv(predictions_file)

    # returning predictions df
    return predictions_df


def get_rmse(test_df: DataFrame,
             predictions_df: DataFrame
             ) -> float:
    """
    Given a test and predictions
    dfs, calculates metrics and
    returns RMSE.
    """
    # joining dfs by crop_name
    print('joining dfs...')
    joined_df = merge(left=test_df,
                      right=predictions_df,
                      on='crop_name')
    print(joined_df)

    # adding rmse cols
    print('adding rmse cols...')
    joined_df['error'] = joined_df['class'] - joined_df['prediction']
    joined_df['squared_error'] = joined_df['error'] * joined_df['error']

    # getting mean squared error
    mse = joined_df['squared_error'].mean()

    # getting root mean squared error
    rmse = sqrt(mse)

    # returning rmse
    return rmse


def nii_regression_test(dataset_file: str,
                        predictions_file: str
                        ) -> None:
    """
    Given a path to dataset df, and
    a path to a file containing test
    data predictions, prints metrics
    on console.
    """
    # getting test df
    print('getting test df...')
    test_df = get_test_df(dataset_file=dataset_file)

    # getting images num
    images_num = len(test_df)

    # getting predictions df
    print('getting predictions df...')
    predictions_df = get_predictions_df(predictions_file=predictions_file)

    # getting metrics df
    print('getting metrics df...')
    rmse = get_rmse(test_df=test_df,
                    predictions_df=predictions_df)

    # printing metrics on console
    print('printing metrics...')
    f_string = f'---Metrics Results---\n'
    f_string += f'Test images num: {images_num}\n'
    f_string += f'RMSE: {rmse}'
    print(f_string)

    # printing execution message
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting dataset file param
    dataset_file = args_dict['dataset_file']

    # predictions file param
    predictions_file = args_dict['predictions_file']

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
