# regression test module

print('initializing...')  # noqa

# Code destined to testing
# regression neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from math import sqrt
from os import environ
from pandas import merge
from pandas import read_csv
from pandas import DataFrame
from seaborn import scatterplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import run_anova_test
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

# setting tensorflow warnings off
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def get_errors_df(test_df: DataFrame,
                  predictions_df: DataFrame
                  ) -> DataFrame:
    """
    Given a test and predictions
    dfs, calculates metrics and
    returns errors df.
    """
    # joining dfs by crop_name
    joined_df = merge(left=test_df,
                      right=predictions_df,
                      on='crop_name')

    # adding error cols
    joined_df['error'] = joined_df['prediction'] - joined_df['class']
    joined_df['squared_error'] = joined_df['error'] * joined_df['error']
    joined_df['absolute_error'] = joined_df['error'].abs()
    joined_df['relative_error'] = joined_df['absolute_error'] / joined_df['class']

    # returning errors df
    return joined_df


def get_error_mean(df: DataFrame,
                   error_col: str
                   ) -> float:
    """
    Given an errors data frame,
    and an error col, returns respective
    column mean value.
    """
    # getting current error col values
    error_col_values = df[error_col]

    # getting current col mean
    col_mean = error_col_values.mean()

    # returning mean value
    return col_mean


def plot_correlations(df: DataFrame,
                      r_squared: float
                      ) -> None:
    """
    Given an errors data frame,
    and r-squared value, displays
    plot on screen.
    """
    # plotting scatterplot
    scatterplot(data=df,
                x='class',
                y='prediction',
                hue='relative_error')

    # setting plot title
    title = f'Correlation plot (class/pred R2: {r_squared})'
    plt.title(title)

    # adjusting layout
    plt.tight_layout()

    # showing plot
    plt.show()

    # closing plot
    plt.close()


def regression_test(dataset_file: str,
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

    # getting errors df
    print('getting errors df...')
    errors_df = get_errors_df(test_df=test_df,
                              predictions_df=predictions_df)
    print(errors_df)
    errors_df = errors_df[errors_df['class'] <= 20]
    print(errors_df)

    # calculating metrics
    print('calculating metrics...')
    mae = get_error_mean(df=errors_df,
                         error_col='absolute_error')
    mre = get_error_mean(df=errors_df,
                         error_col='relative_error')
    mse = get_error_mean(df=errors_df,
                         error_col='squared_error')
    rmse = sqrt(mse)
    r_squared = run_anova_test(df=errors_df,
                               col_real='class',
                               col_pred='prediction')

    # printing metrics on console
    print('printing metrics...')
    f_string = f'---Metrics Results---\n'
    f_string += f'Test images num: {images_num}\n'
    f_string += f'MAE: {mae}\n'
    f_string += f'MRE: {mre}\n'
    f_string += f'MSE: {mse}\n'
    f_string += f'RMSE: {rmse}\n'
    f_string += f'R2: {r_squared}'
    print(f_string)

    # plotting correlations plot
    plot_correlations(df=errors_df,
                      r_squared=r_squared)

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

    # running regression_test function
    regression_test(dataset_file=dataset_file,
                    predictions_file=predictions_file)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
