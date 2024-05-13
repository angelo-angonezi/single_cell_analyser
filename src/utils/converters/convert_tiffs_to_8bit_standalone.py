# image format converter (* -> 8bit)

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os import listdir
from sys import stdout
from cv2 import imread
from cv2 import imwrite
from os.path import join
from numpy import ndarray
from numpy import uint8 as np_uint8
from argparse import ArgumentParser
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert images to 8bit module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    i_help = 'defines path to folder containing input (16-bit .tif) images'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=i_help)

    # output folder param
    o_help = 'defines path to folder which will contain output (8-bit) images'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help=o_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def spacer(char: str = '_',
           reps: int = 50
           ) -> None:
    """
    Given a char and a number of reps,
    prints a "spacer" string assembled
    by multiplying char by reps.
    :param char: String. Represents a character to be used
    as basis for spacer.
    :param reps: Integer. Represents number of character's
    repetitions used in spacer.
    :return: None.
    """
    # defining spacer string
    spacer_str = char * reps

    # printing spacer string
    print(spacer_str)


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
    :param string: String. Represents a string to be printed on console.
    :param index: Integer. Represents an iterable's index.
    :param total: Integer. Represents an iterable's total.
    :return: None.
    """
    # checking whether index is last

    # if current element is last
    if index == total:

        # printing string
        print(string)

    # if current element is not last
    else:

        # flushing string
        flush_string(string)


def enter_to_continue():
    """
    Waits for user input ("Enter")
    and once press, continues to run code.
    """
    # defining enter_string
    enter_string = f'press "Enter" to continue'

    # waiting for user input
    input(enter_string)


def print_progress_message(base_string: str,
                           index: int,
                           total: int
                           ) -> None:
    """
    Given a base string (containing keywords #INDEX#
    and #TOTAL#) an index and a total (integers),
    prints execution message, substituting #INDEX#
    and #TOTAL# keywords by respective integers.
    !!!Useful for FOR loops execution messages!!!
    :param base_string: String. Represents a base string.
    :param index: Integer. Represents an execution index.
    :param total: Integer. Represents iteration maximum value.
    :return: None.
    """
    # getting percentage progress
    progress_ratio = index / total
    progress_percentage = progress_ratio * 100
    progress_percentage_round = round(progress_percentage)

    # assembling progress string
    progress_string = base_string.replace('#INDEX#', str(index))
    progress_string = progress_string.replace('#TOTAL#', str(total))
    progress_string += '...'
    progress_string += f' ({progress_percentage_round}%)'
    progress_string += '     '

    # showing progress message
    flush_or_print(string=progress_string,
                   index=index,
                   total=total)


def print_execution_parameters(params_dict: dict) -> None:
    """
    Given a list of execution parameters,
    prints given parameters on console,
    such as:
    '''
    --Execution parameters--
    input_folder: /home/angelo/Desktop/ml_temp/imgs/
    output_folder: /home/angelo/Desktop/ml_temp/overlays/
    '''
    :param params_dict: Dictionary. Represents execution parameters names and values.
    :return: None.
    """
    # defining base params_string
    params_string = f'--Execution parameters--'

    # iterating over parameters in params_list
    for dict_element in params_dict.items():

        # getting params key/value
        param_key, param_value = dict_element

        # adding 'Enter' to params_string
        params_string += '\n'

        # getting current param string
        current_param_string = f'{param_key}: {param_value}'

        # appending current param string to params_string
        params_string += current_param_string

    # printing final params_string
    spacer()
    print(params_string)
    spacer()


def get_specific_files_in_folder(path_to_folder: str,
                                 extension: str
                                 ) -> list:
    """
    Given a path to a folder, returns a list containing
    all files in folder that match given extension.
    :param path_to_folder: String. Represents a path to a folder.
    :param extension: String. Represents a specific file extension.
    :return: List[str]. Represents all files that match extension in given folder.
    """
    # getting all files in folder
    all_files_in_folder = listdir(path_to_folder)

    # getting specific files
    files_in_dir = [file                          # getting file
                    for file                      # iterating over files
                    in all_files_in_folder        # in input folder
                    if file.endswith(extension)]  # only if file matches given extension

    # sorting list
    files_in_dir = sorted(files_in_dir)

    # returning list
    return files_in_dir


def convert_img_scale(img: ndarray,
                      target_type_min: int,
                      target_type_max: int,
                      target_type: type):
    # getting current image min/max
    img_min = img.min()
    img_max = img.max()

    # getting conversion factors
    a = (target_type_max - target_type_min) / (img_max - img_min)
    b = target_type_max - a * img_max

    # converting image
    converted_img = (a * img + b).astype(target_type)

    # returning converted image
    return converted_img


def convert_single_file(input_file_path: str,
                        output_file_path: str
                        ) -> None:
    """
    Given a path to an image, opens image and
    saves in given output path, converted to 8-bit.
    """
    # opening image
    img = imread(input_file_path, -1)

    # converting image to 8bit
    img_8bit = convert_img_scale(img=img,
                                 target_type_min=0,
                                 target_type_max=256,
                                 target_type=np_uint8)

    # saving image in output folder as 8bit
    imwrite(output_file_path,
            img_8bit)


def convert_multiple_files(input_folder_path: str,
                           output_folder_path: str
                           ) -> None:
    """
    Given a path to a folder containing images,
    converts images to 8bit format, saving them
    in output folder.
    """
    # getting files in input folder
    files = get_specific_files_in_folder(path_to_folder=input_folder_path,
                                         extension='.tif')  # since only tif supports 16-bit
    files_num = len(files)

    # iterating over files in input folder
    for file_index, file in enumerate(files, 1):

        # printing execution message
        f_string = f'converting image #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=file_index,
                               total=files_num)

        # getting input file path
        input_file_path = join(input_folder_path, file)

        # getting output file path
        output_file_path = join(output_folder_path, file)

        # converting image
        convert_single_file(input_file_path=input_file_path,
                            output_file_path=output_file_path)

    # printing execution message
    f_string = f'all {files_num} files converted!'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_folder = args_dict['input_folder']

    # getting output path
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running multiple converter function
    convert_multiple_files(input_folder_path=input_folder,
                           output_folder_path=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
