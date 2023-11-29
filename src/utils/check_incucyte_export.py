# check incucyte export module

print('initializing...')  # noqa

# Code destined to checking incucyte
# exported images.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'check incucyte export (if phase/red images match)'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        type=str,
                        help='defines path to folder containing phase/red subfolders',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def phase_matches_red(phase_images_folder: str,
                      red_images_folder: str
                      ) -> bool:
    """
    Given a path to phase/red folders, returns True
    if all images contained in folders match, and
    False otherwise.
    """
    # getting images in phase folder
    phase_images = get_specific_files_in_folder(path_to_folder=phase_images_folder,
                                                extension='.jpg')

    # getting images in red folder
    red_images = get_specific_files_in_folder(path_to_folder=red_images_folder,
                                              extension='.tif')

    # getting image names
    phase_image_names = [f.replace('.jpg', '') for f in phase_images]
    red_image_names = [f.replace('.tif', '') for f in red_images]

    # getting image numbers
    phase_images_num = len(phase_image_names)
    red_images_num = len(red_image_names)

    # printing execution message
    f_string = f'{phase_images_num} images [.jpg] found in phase folder.\n'
    f_string += f'{red_images_num} images [.tif] found in red folder.'
    spacer()
    print(f_string)

    # checking whether image names match
    match_bool = (phase_image_names == red_image_names)

    # returning boolean value
    return match_bool


def check_incucyte_export(input_folder: str) -> None:
    """
    Given a path to a folder containing phase/red subfolders,
    checks whether all images have respective equivalents in
    each folder, printing execution message as output.
    """
    # getting phase/red subfolder
    phase_images_folder = join(input_folder, 'phase')
    red_images_folder = join(input_folder, 'red')

    # getting match bool
    images_match = phase_matches_red(phase_images_folder=phase_images_folder,
                                     red_images_folder=red_images_folder)

    # checking whether images match
    if images_match:

        # defining execution message
        e_string = 'All images match!'

    else:

        # defining execution message
        e_string = 'Not all images match!\n'
        e_string += 'Please, check incucyte export folders.'

    # printing execution message
    print(e_string)

######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting input folder param
    input_folder = args_dict['input_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running check_incucyte_export function
    check_incucyte_export(input_folder=input_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
