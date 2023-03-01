# analyse compare annotations module

print('initializing...')  # noqa

# Code destined to analysing "compare_annotations.py" module
# output (a csv file containing precision and recall data)

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
import pandas as pd
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from seaborn import lineplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import spacer
print('all required libraries successfully imported.')  # noqa

# next line prevents "SettingWithCopyWarning" pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

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


def plot_prec_rec_curves(df: DataFrame) -> None:
    """
    Given a data frame containing precision and
    recall data, plots precision-recall curves
    coloring by evaluator.
    :param df: DataFrame. Represents a detections data frame.
    :return: None.
    """
    # renaming hue column
    df_cols = df.columns
    df_cols = [f.replace('ann2', 'modelDT')
               for f
               in df_cols]
    df.columns = df_cols

    # defining hue order
    hue_order = ['model03',
                 'model05',
                 'model07']

    # plotting data
    lineplot(data=df,
             x='rec',
             y='prec',
             hue='modelDT',
             hue_order=hue_order)

    # setting title/axis names
    plt.title('Precision-Recall curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # showing plot
    plt.show()


def plot_f1_score_curve(df: DataFrame) -> None:
    """
    Given a data frame containing F1-Score data,
    plots line plot of F1-Score by thresholds,
    coloring by evaluator.
    :param df: DataFrame. Represents a F1-Score data frame.
    :return: None.
    """
    # renaming hue column
    df_cols = df.columns
    df_cols = [f.replace('ann2', 'modelDT')
               for f
               in df_cols]
    df.columns = df_cols

    # defining hue order
    hue_order = ['model03',
                 'model05',
                 'model07']

    # plotting data
    lineplot(data=df,
             x='th',
             y='f1_score',
             hue='modelDT',
             hue_order=hue_order)

    # setting title/axis names
    plt.title('F1-Score curves')
    plt.xlabel('IoU threshold')
    plt.ylabel('F1-Score')

    # showing plot
    plt.show()


def analyse_compare_annotations_output(input_file: str,
                                       output_folder: str
                                       ) -> None:
    """
    Given an input file (compare_annotations.py output),
    generates precision-recall curves, and saves analysis
    dfs in given output folder.
    :param input_file: String. Represents a path to a file.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # reading input file
    print('reading input df...')
    input_df = read_csv(input_file)

    # cleaning df from duplicates
    print('cleaning df from duplicates...')
    clean_df = clear_duplicates_from_df(df=input_df)

    # filtering df for fornmaVSmodels results only
    print('filtering df for fornmaVSmodels results only...')
    filtered_df = clean_df[clean_df['ann1'] == 'fornma']

    # adding F1-Scores to df
    print('adding F1-Scores to df...')
    f1_score_df = add_f1_score_column_to_df(df=filtered_df)

    # saving F1-Scores df
    print('saving F1-Scores df...')
    f1_score_df_save_path = join(output_folder,
                                 'f1_scores_df.csv')
    f1_score_df.to_csv(f1_score_df_save_path,
                       index=False)
    print('saved F1-Scores df in output folder.')

    # plotting precision-recall curves
    print('plotting precision-recall curves...')
    plot_prec_rec_curves(df=f1_score_df)

    # plotting F1-Score curve
    print('plotting F1-Score curve...')
    plot_f1_score_curve(df=f1_score_df)

    # printing execution message
    print('analysis complete.')

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

    # running analyse_compare_annotations_output function
    analyse_compare_annotations_output(input_file=input_file,
                                       output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
