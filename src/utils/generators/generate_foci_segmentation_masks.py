# generate foci segmentation masks module

print('initializing...')  # noqa

# TODO: check if this code will really be necessary and update it!
# Code destined to generating foci
# segmentation masks based on nuclei crops.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imwrite
from os.path import join
from cv2 import convertScaleAbs
from numpy import uint8 as np_uint8
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import load_grayscale_img
from src.utils.aux_funcs import print_progress_message
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
    description = 'generate foci segmentation masks module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to folder containing fluorescent crops (8-bit)')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder')

    # pixel intensity param
    parser.add_argument('-p', '--min-pixel-intensity',
                        dest='min_pixel_intensity',
                        required=True,
                        help='defines pixel intensity threshold to be used (int 0-255)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def generate_foci_segmentation_mask(input_path: str,
                                    output_path: str,
                                    min_pixel_intensity: int
                                    ) -> None:
    """
    Given a path to a fluorescence crop,
    creates segmentation mask based on
    given min pixel intensity, saving
    binary image to given output path.
    """
    # reading image as grayscale
    segmentation_mask = load_grayscale_img(image_path=input_path)

    # changing image contrast
    # segmentation_mask = convertScaleAbs()

    # converting image to binary
    segmentation_mask[segmentation_mask < min_pixel_intensity] = 0
    segmentation_mask[segmentation_mask >= min_pixel_intensity] = 255

    # converting int type
    segmentation_mask = segmentation_mask.astype(np_uint8)

    # saving current segmentation mask
    imwrite(output_path,
            segmentation_mask)


def generate_foci_segmentation_masks(input_folder: str,
                                     images_extension: str,
                                     output_folder: str,
                                     min_pixel_intensity: int
                                     ) -> None:
    """
    Given a path to a folder containing
    fluorescent single nucleus crops,
    generates foci segmentation masks,
    based on given min pixel intensity
    value, saving results to output folder.
    """
    # getting files in input folder
    files = get_specific_files_in_folder(path_to_folder=input_folder,
                                         extension=images_extension)
    files_num = len(files)

    # iterating over files
    for file_index, file in enumerate(files, 1):

        # printing progress message
        base_string = 'generating segmentation mask for crop #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=file_index,
                               total=files_num)

        # getting current image input/output paths
        input_path = join(input_folder,
                          file)
        output_path = join(output_folder,
                           file)

        # generating current image foci segmentation mask
        generate_foci_segmentation_mask(input_path=input_path,
                                        output_path=output_path,
                                        min_pixel_intensity=min_pixel_intensity)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting min pixel intensity
    min_pixel_intensity = int(args_dict['min_pixel_intensity'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_autophagy_dfs function
    generate_foci_segmentation_masks(input_folder=input_folder,
                                     images_extension=images_extension,
                                     output_folder=output_folder,
                                     min_pixel_intensity=min_pixel_intensity)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
