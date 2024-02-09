# generate segmentation masks module

print('initializing...')  # noqa

# Code destined to generating segmentation
# masks based on BRAIND detections.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imwrite
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from numpy import uint8 as np_uint8
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_segmentation_mask
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
    description = 'generate segmentation masks module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # detection file param
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help='defines path to csv file containing model detections')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder')

    # expansion ratio param
    parser.add_argument('-er', '--expansion-ratio',
                        dest='expansion_ratio',
                        help='defines ratio of expansion of width/height to generate larger-than-orig-nucleus crops',
                        required=False,
                        default=1.0)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def create_segmentation_masks(df: DataFrame,
                              output_folder: str,
                              expansion_ratio: float
                              ) -> None:
    """
    Given a detections data frame,
    creates segmentation masks, and
    saves them in given output folder.
    """
    # grouping df by image
    image_groups = df.groupby('img_file_name')

    # getting number of images
    images_num = len(image_groups)

    # defining starter for current_img_index
    current_img_index = 1

    # iterating over image groups
    for image_name, image_group in image_groups:

        # printing execution message
        base_string = 'creating segmentation mask for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_img_index,
                               total=images_num)

        # defining current image save name/path
        save_name = f'{image_name}.tif'
        save_path = join(output_folder,
                         save_name)

        # generating segmentation mask for current image
        segmentation_mask = get_segmentation_mask(df=image_group,
                                                  style='ellipse',
                                                  expansion_ratio=expansion_ratio)

        # converting image to binary
        segmentation_mask[segmentation_mask > 0] = 255

        # converting int type
        segmentation_mask = segmentation_mask.astype(np_uint8)

        # saving current segmentation mask
        imwrite(save_path,
                segmentation_mask)

        # updating current_img_index
        current_img_index += 1


def generate_segmentation_masks(detections_file: str,
                                output_folder: str,
                                expansion_ratio: float
                                ) -> None:
    """
    Given paths to model detections file,
    creates segmentation masks based on
    OBBs info, saving results in given
    output folder.
    """
    # reading detections file
    print('reading detections file...')
    detections_df = read_csv(detections_file)

    # generating segmentation masks
    create_segmentation_masks(df=detections_df,
                              output_folder=output_folder,
                              expansion_ratio=expansion_ratio)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting detections file
    detections_file = args_dict['detection_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting expansion ratio
    expansion_ratio = args_dict['expansion_ratio']
    expansion_ratio = float(expansion_ratio)

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_autophagy_dfs function
    generate_segmentation_masks(detections_file=detections_file,
                                output_folder=output_folder,
                                expansion_ratio=expansion_ratio)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
