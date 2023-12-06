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
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_image_confluence
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

SEED = 53
TEST_SIZE = 0.3

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

    # lines and treatment file param
    parser.add_argument('-lt', '--lines-treatment-file',
                        dest='lines_treatment_file',
                        required=True,
                        help='defines path to csv file containing info on cell lines and treatments.')

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


def get_experiment_well_df(df: DataFrame,
                           experiment: str,
                           well: str
                           ) -> DataFrame:
    """
    Given a data frame, an experiment name
    and a well, returns df filtered by given
    experiment and well.
    """
    # filtering df by experiment name
    experiment_df = df[df['experiment'] == experiment]

    # filtering df by well
    wells_df = experiment_df[experiment_df['well'] == well]

    # getting row
    row = wells_df.iloc[0]

    # returning filtered df row
    return row


def get_image_df(image_name: str,
                 image_group: DataFrame,
                 lines_treatment_df: DataFrame
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

    # getting current lines and treatments df row
    current_lines_treatments_df_row = get_experiment_well_df(df=lines_treatment_df,
                                                             experiment=current_experiment,
                                                             well=current_well)

    # getting current author
    current_author = current_lines_treatments_df_row['author']

    # getting current image cell line
    current_cell_line = current_lines_treatments_df_row['cell_line']

    # getting current image treatment
    current_treatment = current_lines_treatments_df_row['treatment']

    # getting current image confluence
    current_confluence = get_image_confluence(df=image_group,
                                              style='ellipse')

    # assembling current image dict
    current_dict = {'img_name': image_name,
                    'experiment': current_experiment,
                    'author': current_author,
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


def get_base_dataset_df(annotations_df: DataFrame,
                        lines_treatment_df: DataFrame
                        ) -> DataFrame:
    """
    Docstring.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # grouping df by image
    image_groups = annotations_df.groupby('img_file_name')

    # getting images number
    images_num = len(image_groups)

    # defining starter for image index
    image_index = 1

    # iterating over image groups
    for image_name, image_group in image_groups:

        # defining progress message
        progress_string = 'getting info on image #INDEX# of #TOTAL#'

        # printing progress message
        print_progress_message(base_string=progress_string,
                               index=image_index,
                               total=images_num)

        # converting image name to string
        image_name = str(image_name)

        # getting current image df
        current_df = get_image_df(image_name=image_name,
                                  image_group=image_group,
                                  lines_treatment_df=lines_treatment_df)

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
                                    lines_treatment_file: str,
                                    output_path: str
                                    ) -> None:
    """
    Docstring.
    """
    # printing execution message
    print('reading input files...')

    # reading annotations file
    annotations_df = read_csv(annotations_file_path)

    # reading lines treatment file
    lines_treatment_df = read_csv(lines_treatment_file)

    # getting base df
    print('getting base df...')
    base_df = get_base_dataset_df(annotations_df=annotations_df,
                                  lines_treatment_df=lines_treatment_df)

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

    # getting annotations file param
    annotations_file_path = args_dict['annotations_file']

    # getting lines and treatment file param
    lines_treatment_file = args_dict['lines_treatment_file']

    # getting output path param
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running create_dataset_description_file function
    create_dataset_description_file(annotations_file_path=annotations_file_path,
                                    lines_treatment_file=lines_treatment_file,
                                    output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
