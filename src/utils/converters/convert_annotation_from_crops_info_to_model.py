# annotation format converter (crops info csv to model csv)

# annotation format conversion module (from crops info to model)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_mask_area
from src.utils.aux_funcs import get_axis_ratio
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_pixel_intensity
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert annotations from crops info to model"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input path param
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help='defines path to crops info df (.csv) file')

    # class col param
    parser.add_argument('-c', '--class-col',
                        dest='class_col',
                        required=True,
                        help='defines column to be used as "class" column')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output model format (.csv) file')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def convert_single_file(input_csv_file_path: str,
                        class_col: str,
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
        img_file_name = row_data['img_name']
        cx = row_data['cx']
        cy = row_data['cy']
        width = row_data['width']
        height = row_data['height']
        angle = row_data['angle']
        current_class = row_data[class_col]
        detection_threshold = 1.0

        # creating current obb dict
        current_obb_dict = {'img_file_name': img_file_name,
                            'detection_threshold': detection_threshold,
                            'cx': cx,
                            'cy': cy,
                            'width': width,
                            'height': height,
                            'angle': angle,
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

    # getting class col
    class_col = args_dict['class_col']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running converter function
    convert_single_file(input_csv_file_path=input_file,
                        class_col=class_col,
                        output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
