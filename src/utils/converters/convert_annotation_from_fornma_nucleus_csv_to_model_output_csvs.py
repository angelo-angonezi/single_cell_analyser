# annotation format converter (fornma csv to model output csv)

# annotation format conversion module (from fornma output to model output format)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
from sys import stdout
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser

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

    # foci threshold param
    parser.add_argument('-t', '--foci-threshold',
                        dest='foci_threshold',
                        required=True,
                        help='defines threshold for "HighDamage" and "LowDamage" class definition')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining auxiliary functions

def flush_string(string: str) -> None:
    """
    Given a string, writes and flushes it in the console using
    sys library, and resets cursor to the start of the line.
    (writes N backspaces at the end of line, where N = len(string)).
    :param string: String. Represents a message to be written in the console.
    :return: None.
    """
    # getting string length
    string_len = len(string)

    # creating backspace line
    backspace_line = '\b' * string_len

    # writing string
    stdout.write(string)

    # flushing console
    stdout.flush()

    # resetting cursor to start of the line
    stdout.write(backspace_line)


def flush_or_print(string: str,
                   index: int,
                   total: int
                   ) -> None:
    """
    Given a string, prints string if index
    is equal to total, and flushes it on console
    otherwise.
    !(useful for progress tracking/progress bars)!
    :param string: String. Represents a string to be printed on console;
    :param index: Integer. Represents an iterable's index;
    :param total: Integer. Represents an iterable's total.
    :return: None.
    """
    # checking whether index is last
    if index == total:  # current element is last

        # printing string
        print(string)

    else:  # current element is not last

        # flushing string
        flush_string(string)


def convert_single_file(input_csv_file_path: str,
                        output_path: str,
                        foci_threshold: int
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
        percentage_progress = row_index / rows_num
        percentage_progress *= 100
        percentage_progress = round(percentage_progress)
        f_string = f'getting info on OBB {row_index} of {rows_num}... ({percentage_progress}%)'
        flush_or_print(string=f_string,
                       index=row_index,
                       total=rows_num)

        # getting file name
        file_name = row_data['Image_name_53bp1']
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

        # getting current nucleus foci count
        foci_count = row_data['Total_foci_53bp1']
        foci_count = int(foci_count)

        # defining current class based on foci count
        current_class = 'HighDamage' if foci_count > foci_threshold else 'LowDamage'

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

    # getting foci threshold
    foci_threshold = args_dict['foci_threshold']
    foci_threshold = int(foci_threshold)

    # running multiple converter function
    convert_single_file(input_csv_file_path=input_file,
                        output_path=output_path,
                        foci_threshold=foci_threshold)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
