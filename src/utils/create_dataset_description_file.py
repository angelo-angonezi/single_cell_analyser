# create data set description file module

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
from src.utils.aux_funcs import get_image_confluence
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

SEED = 53
TEST_SIZE = 0.3
TREATMENT_DICT = {'A172_ERK_53BP1_11_08_22_apos_tratamento':
                      {'A1': 'TMZ',
                       'A2': 'TMZ',
                       'A3': 'TMZ',
                       'B1': 'CTR',
                       'B2': 'CTR',
                       'B3': 'CTR'},
                  'A172_H2B_U251_H2B_ActD_TMZ_daphne':
                      {'B2': 'TMZ',
                       'B3': 'TMZ',
                       'B4': 'TMZ',
                       'B5': 'TMZ',
                       'B6': 'TMZ',
                       'C2': 'CTR',
                       'C3': 'CTR',
                       'C4': 'CTR',
                       'C5': 'CTR',
                       'C6': 'CTR'}
                  }
CELL_LINES_DICT = {'A172_ERK_53BP1_11_08_22_apos_tratamento':
                       {'A1': 'A172',
                        'A2': 'A172',
                        'A3': 'A172',
                        'B1': 'A172',
                        'B2': 'A172',
                        'B3': 'A172'},
                   'A172_H2B_U251_H2B_ActD_TMZ_daphne':
                       {'B2': 'A172',
                        'B3': 'A172',
                        'B4': 'A172',
                        'B5': 'A172',
                        'B6': 'A172',
                        'C2': 'U251',
                        'C3': 'U251',
                        'C4': 'U251',
                        'C5': 'U251',
                        'C6': 'U251'}
                   }

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


def get_image_df(image_name: str,
                 image_group: DataFrame
                 ) -> DataFrame:
    """
    Given an image name and group,
    returns given image base data
    set data frame.
    """
    # removing current image extension
    image_name = image_name.replace('.tif', '')

    # getting current image cell count
    cell_count = len(image_group)

    # getting current image name split
    image_name_split = image_name.split('_')

    # getting image experiment string list
    experiment_split = image_name_split[:-4]

    # getting image experiment
    current_experiment = '_'.join(experiment_split)

    # getting current image well
    current_well = image_name_split[-4]

    # getting current image field
    current_field = image_name_split[-3]

    # getting current image cell line
    current_cell_line_dict = CELL_LINES_DICT[current_experiment]
    current_cell_line = current_cell_line_dict[current_well]

    # getting current image treatment
    current_treatment_dict = TREATMENT_DICT[current_experiment]
    current_treatment = current_treatment_dict[current_well]

    # getting current image confluence
    current_confluence = get_image_confluence(df=image_group,
                                              style='ellipse')

    # assembling current image dict
    current_dict = {'img_name': image_name,
                    'experiment': current_experiment,
                    'well': current_well,
                    'field': current_field,
                    'cell_line': current_cell_line,
                    'treatment': current_treatment,
                    'cell_count': cell_count,
                    'confluence': current_confluence}

    # assembling current image df
    current_df = DataFrame(current_dict,
                           index=[0])

    # returning current image df
    return current_df


def get_base_dataset_df(input_file: str) -> DataFrame:
    """
    Docstring.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # reading annotations file
    annotations_df = read_csv(input_file)

    # grouping df by image
    image_groups = annotations_df.groupby('img_file_name')

    # getting images number
    images_num = len(image_groups)

    # defining starter for image index
    image_index = 1

    # iterating over image groups
    for image_name, image_group in image_groups:

        # defining progress message
        progress_string = 'analysing image #INDEX# of #TOTAL#'

        # printing progress message
        print_progress_message(base_string=progress_string,
                               index=image_index,
                               total=images_num)

        # converting image name to string
        image_name = str(image_name)

        # getting current image df
        current_df = get_image_df(image_name=image_name,
                                  image_group=image_group)

        # appending current df to dfs list
        dfs_list.append(current_df)

        # updating image index
        image_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def add_dataset_col(df: DataFrame,
                    test_size: float
                    ) -> None:
    """
    Docstring.
    """
    # TODO: write this function!
    #  group by cell line, treatment and confluence,
    #  and add train/test to dataset column according
    #  to test size.
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
    print(base_df)

    # adding dataset (train/test) col
    add_dataset_col(df=base_df,
                    test_size=TEST_SIZE)
    print(base_df)

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
