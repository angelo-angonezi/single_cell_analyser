# create data set splits file module

print('initializing...')  # noqa

# Code destined to generating project's data set split
# file, in train/test images with stratification
# between cell lines, treatments and confluences.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from random import seed as set_seed
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_image_confluence
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

SEED = 53
TEST_SIZE = 0.3
CONFLUENCE_THRESHOLDS = 0.05

# setting seed (so that all executions result in same sample)
set_seed(SEED)

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'create data set splits file module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # annotations file param
    parser.add_argument('-d', '--dataset-description-file',
                        dest='dataset_description_file',
                        required=True,
                        help='defines path to data set description file (.csv)')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output csv.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_confluence_group(confluence: float,
                         confluence_mean: float,
                         confluence_std: float
                         ) -> str:
    """
    Given a confluence (PERCENTAGE) value,
    returns a confluence group.
    """
    # TODO: update this function
    # checking whether confluence surpasses minimum value
    if confluence > 0.5:

        # setting confluence group as High
        confluence_group = 'High'

    # setting confluence group as Low
    confluence_group = 'Low'

    # returning confluence_group
    return confluence_group


def add_confluence_group_col(df: DataFrame) -> None:
    """
    Docstring.
    """
    # defining new col name
    col_name = 'confluence_group'

    # adding placeholder "confluence_group" col
    df[col_name] = None

    # adding confluence percentage col
    df['confluence_percentage'] = df['confluence'] * 100

    # getting confluence mean/std
    confluence_mean = df['confluence_percentage'].mean()
    confluence_std = df['confluence_percentage'].std()

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current_row_index
    current_row_index = 1

    # iterating over rows
    for row_index, row_data in df_rows:

        # printing execution message
        base_string = f'adding confluence group to row #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row confluence percentage
        current_confluence_percentage = row_data['confluence_percentage']

        # getting current row confluence group
        current_confluence_group = get_confluence_group(confluence=current_confluence_percentage)

        # updating confluence group col
        df.at[row_index, col_name] = current_confluence_group

        # updating current row index
        current_row_index += 1

    exit()


def add_dataset_col(df: DataFrame,
                    test_size: float
                    ) -> None:
    """
    Docstring.
    """
    # adding placeholder "split" col
    df['split'] = None

    # defining groups
    groups_list = ['cell_line', 'treatment', 'confluence']

    # grouping df
    df_groups = df.groupby(groups_list)

    # iterating over groups
    for df_name, df_group in df_groups:

        # getting df data
        cell_line, treatment, confluence = df_name

        # randomly splitting current group rows
        print(df_name)
        print(df_group)
        exit()



def create_dataset_splits_file(dataset_description_file: str,
                               output_path: str
                               ) -> None:
    """
    Docstring.
    """
    # printing execution message
    print('reading input files...')

    # reading dataset description file
    base_df = read_csv(dataset_description_file)

    # adding confluence group col
    print('adding confluence class col...')
    add_confluence_group_col(df=base_df)

    # adding dataset (train/test) col
    print('adding data split col...')
    add_dataset_col(df=base_df,
                    test_size=TEST_SIZE)

    # saving dataset description df
    base_df.to_csv(output_path,
                   index=False)
    print('saving dataset description df...')

    # printing execution message
    f_string = f'dataset description file saved to: {output_path}'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting dataset description file param
    dataset_description_file = args_dict['dataset_description_file']

    # getting output path param
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running create_dataset_description_file function
    create_dataset_splits_file(dataset_description_file=dataset_description_file,
                               output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
