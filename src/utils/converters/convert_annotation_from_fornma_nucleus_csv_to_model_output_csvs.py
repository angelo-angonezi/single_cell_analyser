# annotation format converter (fornma csv to model output csv)

# annotation format conversion module (from fornma output to model output format)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import print_progress_message
print('all required libraries successfully imported.')  # noqa

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


def convert_single_file(input_csv_file_path: str,
                        output_path: str
                        ) -> None:
    """
    Given a path to a fornma output file containing cell
    nucleus annotations, converts annotations to
    model output format.
    """
    # opening csv file
    fornma_df = read_csv(input_csv_file_path)

    # defining placeholder value for dfs list
    dfs_list = []

    # getting bounding boxes and objects from csv file (lines in table)
    rows = [line for line in fornma_df.iterrows()]
    rows_num = len(rows)

    # iterating over fornma df rows
    for row in rows:

        # getting row index and row data
        row_index, row_data = row

        # correcting index
        row_index += 1

        # flushing/printing execution message
        f_string = f'getting info on OBB #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=row_index,
                               total=rows_num)

        # getting file name
        file_name = row_data['Image_name_red']
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

        # defining current class
        current_class = 'Nucleus'

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

    # concatenating dfs in dfs list
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

    # running multiple converter function
    convert_single_file(input_csv_file_path=input_file,
                        output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
