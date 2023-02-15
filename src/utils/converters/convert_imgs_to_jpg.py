# image format converter (* -> jpg)

######################################################################
# imports

# importing required libraries
from PIL import Image
from os import listdir
from sys import stdout
from os.path import join
from argparse import ArgumentParser

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert images to jpg format module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    i_help = 'defines path to folder containing input images'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=i_help)

    # output folder param
    o_help = 'defines path to folder which will contain output[.jpg] images'
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


def convert_single_file(input_file_path: str,
                        output_file_path: str
                        ) -> None:
    """
    Given a path to an image, opens image and
    saves in given output path, converted to jpg.
    """
    # opening image
    img = Image.open(input_file_path)

    # saving image in output folder as jpg
    img.save(output_file_path)


def convert_multiple_files(input_folder_path: str,
                           output_folder_path: str
                           ) -> None:
    """
    Given a path to a folder containing images,
    converts images to jpg format, saving them
    in output folder.
    """
    # getting files in input folder
    files = [file
             for file
             in listdir(input_folder_path)
             if (file.endswith('.jpg'))
             or (file.endswith('.tif'))]
    files_num = len(files)

    # iterating over files in input folder
    for file_index, file in enumerate(files, 1):

        # printing execution message
        progress_ratio = file_index / files_num
        progress_percentage = progress_ratio * 100
        progress_percentage_round = round(progress_percentage)
        f_string = f'converting image {file_index} of {files_num} ({progress_percentage_round}%)      '
        flush_or_print(string=f_string,
                       index=file_index,
                       total=files_num)

        # getting input file path
        input_file_path = join(input_folder_path, file)

        # getting output file path
        save_name = file.replace('.tif', '.jpg')
        save_name = save_name.replace('.png', '.jpg')
        output_file_path = join(output_folder_path, save_name)

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

    # running multiple converter function
    convert_multiple_files(input_folder_path=input_folder,
                           output_folder_path=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
