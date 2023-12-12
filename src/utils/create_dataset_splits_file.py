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


def add_confluence_group_col(df: DataFrame) -> None:
    """
    Docstring.
    """
    # adding confluence percentage col
    df['confluence_percentage'] = df['confluence'] * 100

    # getting confluence percentage round values
    df['confluence_percentage_round'] = df['confluence_percentage'].round()

    # getting confluence percentage int values
    df['confluence_percentage_int'] = df['confluence_percentage_round'].astype(int)

    # getting confluence percentage str values
    df['confluence_percentage_str'] = df['confluence_percentage_int'].astype(str)

    # getting confluence group values
    df['confluence_group'] = df['confluence_percentage_str'].replace('0', '<1')


def add_dataset_col(df: DataFrame,
                    test_size: float
                    ) -> None:
    """
    Docstring.
    """
    # defining split col name
    split_col_name = 'split'

    # adding placeholder "split" col
    df[split_col_name] = None

    # defining groups
    groups_list = ['cell_line', 'treatment', 'confluence_group']

    # grouping df
    df_groups = df.groupby(groups_list)

    # getting groups num
    groups_num = len(df_groups)

    # printing execution message
    f_string = f'{groups_num} were found based on: {groups_list}'
    print(f_string)

    # iterating over groups
    for df_name, df_group in df_groups:

        # randomly splitting current group rows
        current_test_split = df_group.sample(frac=test_size)
        current_train_split = df_group.drop(current_test_split.index)

        # getting train/test indices
        train_indices = current_train_split.index
        test_indices = current_test_split.index

        # adding split column based on current samples
        for train_index in train_indices:
            df.at[train_index, split_col_name] = 'train'
        for test_index in test_indices:
            df.at[test_index, split_col_name] = 'test'


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
    print('adding confluence group col...')
    add_confluence_group_col(df=base_df)

    # adding dataset (train/test) col
    print('adding data split col...')
    add_dataset_col(df=base_df,
                    test_size=TEST_SIZE)

    # saving dataset description df
    print('saving dataset description df...')
    base_df.to_csv(output_path,
                   index=False)

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
