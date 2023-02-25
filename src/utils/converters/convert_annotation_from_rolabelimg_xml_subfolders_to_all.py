# annotation format converter (rolabelimg xml to all formats)

# annotation format conversion module (from rolabelimg to all formats)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
from os import system
from os import listdir
from os.path import join
from os.path import isdir
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer

#####################################################################
# defining global parameters

PATH_TO_CONVERTER_ROLABELIMG_TO_ALPR = 'Z:\\pycharm_projects\\single_cell_analyser\\src\\utils\\converters\\convert_annotation_from_rolabelimg_xml_to_alpr.py'
PATH_TO_CONVERTER_ALPR_TO_DOTA = 'Z:\\pycharm_projects\\RotationDetection\\dataloader\\dataset\\UFRGS_CELL\\convert_ann2dota.py'
PATH_TO_CONVERTER_DOTA_TO_LUCAS_XML = 'Z:\\pycharm_projects\\RotationDetection\\dataloader\\dataset\\UFRGS_CELL\\txt2xml.py'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert annotations from rolabelimg xml to all formats"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder param
    imgs_help = 'defines images input folder (phase [.jpgs])'
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help=imgs_help)

    # rolabelimg folder param
    rolabelimg_help = 'defines input folder '
    rolabelimg_help += '(folder containing subfolders with rolabelimg[.xml] annotations for each evaluator)'
    parser.add_argument('-r', '--rolabelimg-folder',
                        dest='rolabelimg_folder',
                        required=True,
                        help=rolabelimg_help)

    # alpr folder param
    alpr_help = 'defines output folder '
    alpr_help += '(folder with subfolders that will contain alpr[.txt] annotations for each evaluator)'
    parser.add_argument('-a', '--alpr-folder',
                        dest='alpr_folder',
                        required=True,
                        help=alpr_help)

    # dota folder param
    dota_help = 'defines output folder '
    dota_help += '(folder with subfolders that will contain dota[.txt] annotations for each evaluator)'
    parser.add_argument('-d', '--dota-folder',
                        dest='dota_folder',
                        required=True,
                        help=dota_help)

    # lucas_xml folder param
    lucas_xml_help = 'defines output folder '
    lucas_xml_help += '(folder with subfolders that will contain lucas_xmls[.xml] annotations for each evaluator)'
    parser.add_argument('-l', '--lucas-xmls-folder',
                        dest='lucas_xmls_folder',
                        required=True,
                        help=lucas_xml_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def run_command(command_string: str) -> None:
    """
    Given a string representing a command to be
    run in console, executes command using os.system
    function.
    """
    # printing execution message
    f_string = f'running command "{command_string}"...'
    print(f_string)

    # running command
    system(command=command_string)


def get_subfolders_in_dir(folder_path: str) -> list:
    """
    Given a path to a directory, returns
    all subfolders names in given directory.
    """
    # getting subfolders in dir
    subfolders = [f
                  for f
                  in listdir(folder_path)
                  if isdir(join(folder_path, f))]

    # returning subfolders list
    return subfolders


def convert_files_in_single_dir(subfolder: str,
                                images_folder_path: str,
                                rolabelimg_folder_path: str,
                                alpr_folder_path: str,
                                dota_folder_path: str,
                                lucas_xmls_folder_path: str
                                ) -> None:
    """
    Given a path to a subfolder containing annotations in
    rolabelimg format, converts annotations to all formats.
    """
    # defining subfolder paths (used as execution parameters for commands)
    subfolder_path_rolabelimg = join(rolabelimg_folder_path, subfolder)
    subfolder_path_alpr = join(alpr_folder_path, subfolder)
    subfolder_path_dota = join(dota_folder_path, subfolder)
    subfolder_path_lucas_xml = join(lucas_xmls_folder_path, subfolder)

    # creating rolabelimg to alpr command string
    rolabelimg_to_alpr_command = f"python3 '{PATH_TO_CONVERTER_ROLABELIMG_TO_ALPR}' -i '{subfolder_path_rolabelimg}' -o '{subfolder_path_alpr}'"

    # creating alpr to dota command string
    alpr_to_dota_command = f"python3 '{PATH_TO_CONVERTER_ALPR_TO_DOTA}' '{images_folder_path}' '{subfolder_path_alpr}' '{subfolder_path_dota}'"

    # creating dota to lucas xmls command string
    dota_to_lucas_xml_command = f"python3 '{PATH_TO_CONVERTER_DOTA_TO_LUCAS_XML}' '{subfolder_path_dota}' '{subfolder_path_lucas_xml}' '{images_folder_path}'"

    # running commands
    run_command(command_string=rolabelimg_to_alpr_command)
    run_command(command_string=alpr_to_dota_command)
    run_command(command_string=dota_to_lucas_xml_command)


def convert_files_in_multiple_dirs(images_folder_path: str,
                                   rolabelimg_folder_path: str,
                                   alpr_folder_path: str,
                                   dota_folder_path: str,
                                   lucas_xmls_folder_path: str
                                   ) -> None:
    """
    Given a path to a folder containing multiple subfolders containing
    annotations in rolabelimg format, converts annotations to all formats.
    """
    # getting subfolders in input folder
    subfolders = get_subfolders_in_dir(folder_path=rolabelimg_folder_path)
    subfolders_count = len(subfolders)

    # iterating over subfolders
    for subfolder_index, subfolder in enumerate(subfolders, 1):

        # printing execution message
        f_string = f'converting annotations for evaluator: {subfolder} ({subfolder_index} of {subfolders_count})'
        spacer()
        print(f_string)

        # converting files in current subfolder
        convert_files_in_single_dir(subfolder=subfolder,
                                    images_folder_path=images_folder_path,
                                    rolabelimg_folder_path=rolabelimg_folder_path,
                                    alpr_folder_path=alpr_folder_path,
                                    dota_folder_path=dota_folder_path,
                                    lucas_xmls_folder_path=lucas_xmls_folder_path)

    # printing execution message
    f_string = 'all annotation files converted!'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder
    images_folder = args_dict['images_folder']

    # getting rolabelimg folder
    rolabelimg_folder = args_dict['rolabelimg_folder']

    # getting alpr folder
    alpr_folder = args_dict['alpr_folder']

    # getting dota folder
    dota_folder = args_dict['dota_folder']

    # getting lucas xmls folder
    lucas_xmls_folder = args_dict['lucas_xmls_folder']

    # printing execution message
    f_string = f'--Execution Parameters--\n'
    f_string += f'images folder: {images_folder}\n'
    f_string += f'rolabelimg folder: {rolabelimg_folder}\n'
    f_string += f'alpr folder: {alpr_folder}\n'
    f_string += f'dota folder: {dota_folder}\n'
    f_string += f'lucas xmls folder: {lucas_xmls_folder}'
    spacer()
    print(f_string)
    spacer()

    # waiting for user input
    e_string = 'Press "Enter" to continue'
    input(e_string)

    # running multiple converter function
    convert_files_in_multiple_dirs(images_folder_path=images_folder,
                                   rolabelimg_folder_path=rolabelimg_folder,
                                   alpr_folder_path=alpr_folder,
                                   dota_folder_path=dota_folder,
                                   lucas_xmls_folder_path=lucas_xmls_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module