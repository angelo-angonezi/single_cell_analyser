# annotation format converter (crops info csv to fornma csv)

# annotation format conversion module (from crops info to fornma)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import concat
from pandas import Series
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_pixel_intensity
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

IMAGE_NAME_COL = 'Image_name_merge'
RED_MEAN_COL = 'Mean_red'
GREEN_MEAN_COL = 'Mean_green'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert annotations from crops info to fornma"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input path param
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help='defines path to crops info df (.csv) file')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output fornma format (.csv) file')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_phenotype(row_data: Series,
                  phenotype: str,
                  phenotype_col: str):
    """
    Given a row data, phenotype and a
    phenotype column, returns respective
    "phenotype value".
    """
    # defining placeholder value for phenotype value
    phenotype_value = None

    # trying to access column value
    try:

        # getting phenotype col value
        row_value = row_data[phenotype_col]

    # if column does not exist
    except KeyError:

        # printing error message
        e_string = f'Phenotype column "{phenotype_col}" does not exist.'
        e_string += 'Please, check and try again.'
        print(e_string)
        exit()

    # checking phenotype
    if phenotype == 'dna_damage':

        # getting phenotype
        phenotype_value = 'HighDamage' if row_value > FOCI_THRESHOLD else 'LowDamage'

    else:

        # printing error message
        e_string = f'Invalid phenotype: "{phenotype}"'
        e_string += 'Please, check available phenotypes and try again.'
        print(e_string)
        exit()

    # returning phenotype value
    return phenotype_value


def convert_single_file(input_csv_file_path: str,
                        output_path: str
                        ) -> None:
    """
    Given a path to a crops info file,
    converts annotations to fornma format,
    saving results in given output path.
    """
    # opening csv file
    print('reading input file...')
    crops_df = read_csv(input_csv_file_path)

    # defining placeholder value for dfs list
    dfs_list = []

    # getting rows num
    rows_num = len(crops_df)

    # getting fornma df rows
    rows = crops_df.iterrows()

    # defining starter for current_row_index
    current_row_index = 1

    # iterating over fornma df rows
    for row_index, row_data in rows:

        # flushing/printing execution message
        f_string = f'getting info on nucleus #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row cols
        current_img_name = row_data['img_name']
        current_crop_index = row_data['crop_index']
        current_crop_name = row_data['crop_name']
        current_cx = row_data['cx']
        current_cy = row_data['cy']
        current_width = row_data['width']
        current_height = row_data['height']
        current_angle = row_data['angle']
        current_experiment = row_data['experiment']
        current_well = row_data['well']
        current_field = row_data['field']
        current_date = row_data['date']
        current_time = row_data['time']
        current_treatment = row_data['treatment']
        print(current_crop_name)
        exit()

        # getting current crop channel paths
        current_red_path = None
        current_green_path = None

        # getting current crop mean pixel intensities

        # creating current obb dict
        current_obb_dict = {'img_file_name': file_name,
                            'detection_threshold': 1.0,
                            'cx': cx_float,
                            'cy': cy_float,
                            'width': width_float,
                            'height': height_float,
                            'angle': angle_in_degs_float,
                            'class': current_class}

        # creating current obb df
        current_obb_df = DataFrame(current_obb_dict,
                                   index=[0])

        # appending current obb df to dfs list
        dfs_list.append(current_obb_df)

        # updating current_row_index
        current_row_index += 1

    # concatenating dfs in dfs list
    print('assembling final df...')
    final_df = concat(dfs_list)

    # saving final df in output path
    print('saving output file...')
    final_df.to_csv(output_path,
                    index=False)

    # printing execution message
    f_string = f'all fornma results successfully converted!\n'
    f_string += f'results saved at "{output_path}"'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running converter function
    convert_single_file(input_csv_file_path=input_file,
                        output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
