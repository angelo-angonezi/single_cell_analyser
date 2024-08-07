# image format converter (* -> png)

######################################################################
# imports

# importing required libraries
from PIL import Image
from os import listdir
from os.path import join
from argparse import ArgumentParser
from src.utils.aux_funcs import print_progress_message

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert images to png format module"

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
    o_help = 'defines path to folder which will contain output[.png] images'
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


def convert_single_file(input_file_path: str,
                        output_file_path: str
                        ) -> None:
    """
    Given a path to an image, opens image and
    saves in given output path, converted to png.
    """
    # opening image
    img = Image.open(input_file_path)

    # saving image in output folder as png
    img.save(output_file_path)


def convert_multiple_files(input_folder_path: str,
                           output_folder_path: str
                           ) -> None:
    """
    Given a path to a folder containing images,
    converts images to png format, saving them
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
        f_string = f'converting image #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=file_index,
                               total=files_num)

        # getting input file path
        input_file_path = join(input_folder_path, file)

        # getting output file path
        save_name = file.replace('.tif', '.png')
        save_name = save_name.replace('.jpg', '.png')
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
