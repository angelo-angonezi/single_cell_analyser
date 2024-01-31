# annotation format converter (fornma csv to model output csv)

# annotation format conversion module (from fornma output to model output format)
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
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

PHENOTYPE = 'dna_damage'
PHENOTYPE_COL = 'Total_foci_merge'
IMAGE_NAME_COL = 'Image_name_merge'
FOCI_THRESHOLD = 3

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert annotations from fornma output to model output format"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input path param
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help='defines path to file containing fornma[.csv] NUCLEUS results')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines output path[.csv]')

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
    Given a path to a fornma output file containing cell
    nucleus annotations, converts annotations to
    model output format.
    """
    # opening csv file
    print('reading input file...')
    fornma_df = read_csv(input_csv_file_path)

    # defining placeholder value for dfs list
    dfs_list = []

    # getting rows num
    rows_num = len(fornma_df)

    # getting fornma df rows
    rows = fornma_df.iterrows()

    # defining starter for current_row_index
    current_row_index = 1

    # iterating over fornma df rows
    for row_index, row_data in rows:

        # flushing/printing execution message
        f_string = f'getting info on OBB #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=current_row_index,
                               total=rows_num)

        # getting file name
        file_name = row_data[IMAGE_NAME_COL]
        file_name = file_name.replace('.tif', '')

        # getting center x value
        cx_text = row_data['FitEllipse_X']
        cx_float = float(cx_text)

        # getting center y value
        cy_text = row_data['FitEllipse_Y']
        cy_float = float(cy_text)

        # getting width value
        width_text = row_data['FitEllipse_b']
        width_float = float(width_text)

        # getting height value
        height_text = row_data['FitEllipse_a']
        height_float = float(height_text)

        # getting angle value
        angle_in_degs_text = row_data['FitEllipse_angle']
        angle_in_degs_float = float(angle_in_degs_text)

        # defining current class based on foci count
        current_class = get_phenotype(row_data=row_data,
                                      phenotype=PHENOTYPE,
                                      phenotype_col=PHENOTYPE_COL)

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
    input_file = str(input_file)

    # getting output path
    output_path = args_dict['output_path']
    output_path = str(output_path)

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running multiple converter function
    convert_single_file(input_csv_file_path=input_file,
                        output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
