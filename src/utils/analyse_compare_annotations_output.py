# analyse compare annotations module

print('initializing...')  # noqa

# Code destined to analysing "compare_annotations.py" module
# output (a csv file containing precision and recall data)

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import flush_or_print
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "analyse compare_annotations.py output module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input file param
    input_help = 'defines path to compare_annotations.py output[.csv]'
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help=input_help)

    # output file param
    output_help = 'defines output folder (folder that will contain output csvs)'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help=output_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def clear_duplicates_from_df(df: DataFrame) -> DataFrame:
    """
    Given annotators and models names' list,
    and an input data frame, returns data frame
    filtered so that it does not contain duplicate
    lines as follows:
    | ann1   | ann2   | ... | ... |
    | fornma | fornma | ... | ... |
    | model    | model    | ... | ... |
    ...
    """
    # defining keep condition
    keep_condition = (df['ann1'] != df['ann2'])

    # clearing df based on keep condition
    clean_df = df[keep_condition]

    # returning clean df
    return clean_df


def add_f1_score_column_to_df(df: DataFrame) -> DataFrame:
    """
    Given an annotatorsVSmodels variability output
    data frame, computes F1-score based on precision
    and recall columns, returning a new data frame
    containing F1-scores in new column.
    :param df: DataFrame. Represents a detections data frame.
    :return: DataFrame. Represents a detections data frame.
    """
    # adding precision*recall column to df
    df['precision_times_recall'] = df['prec'] * df['rec']

    # adding precision+recall column to df
    df['precision_plus_recall'] = df['prec'] + df['rec']

    # adding div column to df
    df['div'] = df['precision_times_recall'] / df['precision_plus_recall']

    # adding f1_score column to df
    df['f1_score'] = 2 * df['div']

    # returning modified df
    return df


def get_prec_rec_df(df:DataFrame) -> DataFrame:
    """

    :param df: DataFrame. Represents a detections data frame.
    :return: DataFrame. Represents ready-to-plot detections data frame.
    """
    pass


def plot_prec_rec_curves(df: DataFrame) -> None:
    """
    Given a data frame containing precision and
    recall data, plots precision-recall curves
    coloring by evaluator.
    :param df: DataFrame. Represents a detections data frame.
    :return: None.
    """
    pass


######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    f_string = f'--Execution parameters--\n'
    f_string += f'input file: {input_file}\n'
    f_string += f'output folder: {output_folder}'
    spacer()
    print(f_string)
    spacer()
    input('press "Enter" to continue')

    # running

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
