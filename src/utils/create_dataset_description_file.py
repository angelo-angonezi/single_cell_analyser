# generate base imgs info file module

print('initializing...')  # noqa

# Code destined to generating project's data set description
# as well as splitting train/test images with stratification
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
from src.utils.aux_funcs import get_confluences_df
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

SEED = 53
TEST_SIZE = 0.3
TREATMENT_DICT = {'A172_BLABLA': {'A1': 'TMZ',
                                  'B1': 'CTR'}}

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
    description = 'create data set description file module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # annotations file param
    parser.add_argument('-a', '--annotations-file',
                        dest='annotations_file',
                        required=True,
                        help='defines path to fornma nucleus output file (model output format).')

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


def get_base_dataset_df(input_file: str) -> DataFrame:
    """
    Docstring.
    """
    # reading annotations file
    annotations_df = read_csv(input_file)

    # adding confluence column
    confluences_df = get_confluences_df(df=annotations_df,
                                        style='ellipse')

    print(annotations_df)
    print(confluences_df)
    exit()


def add_dataset_col(df: DataFrame,
                    test_size: float
                    ) -> None:
    """
    Docstring.
    """
    pass


def create_dataset_description_file(annotations_file_path: str,
                                    output_path: str
                                    ) -> None:
    """
    Docstring.
    """
    # getting base df
    print('reading input file...')
    base_df = get_base_dataset_df(input_file=annotations_file_path)

    exit()

    # adding dataset (train/test) col
    add_dataset_col(df=base_df,
                    test_size=TEST_SIZE)

    exit()

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

    # getting annotations file param
    annotations_file_path = args_dict['annotations_file']

    # getting output path param
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running create_dataset_description_file function
    create_dataset_description_file(annotations_file_path=annotations_file_path,
                                    output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
