# generate autophagy df module

print('initializing...')  # noqa

# Code destined to generating autophagy
# data frame, based on segmentation masks.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from cv2 import imwrite
from cv2 import cvtColor
from os.path import join
from pandas import concat
from cv2 import contourArea
from cv2 import drawContours
from pandas import DataFrame
from cv2 import boundingRect
from cv2 import findContours
from cv2 import RETR_EXTERNAL
from cv2 import COLOR_GRAY2BGR
from cv2 import CHAIN_APPROX_NONE
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

CELL_MIN_AREA = 100
FOCI_MIN_AREA = 2
COLOR_DICT = {'cell': (0, 0, 255),  # blue
              'foci': (0, 255, 0)}  # green

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate autophagy df module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing original images (8-bit .tif)')

    # cell masks folder param
    parser.add_argument('-c', '--cell-masks-folder',
                        dest='cell_masks_folder',
                        required=True,
                        help='defines path to folder containing cell segmentation masks (macro output)')

    # foci masks folder param
    parser.add_argument('-f', '--foci-masks-folder',
                        dest='foci_masks_folder',
                        required=True,
                        help='defines path to folder containing foci segmentation masks (macro output)')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_contours_df(image_name: str,
                    image_path: str,
                    contour_type: str
                    ) -> DataFrame:
    """
    Given a path to a binary image,
    finds contours and returns data
    frame containing contours coords.
    """
    # reading image
    image = imread(image_path,
                   -1)

    # finding contours in image
    contours, _ = findContours(image, RETR_EXTERNAL, CHAIN_APPROX_NONE)

    # getting number of contours found in image
    contours_num = len(contours)
    contours_num_range = range(contours_num)

    # getting current contours coords
    contours_coords = [boundingRect(contour) for contour in contours]

    # getting current contours areas
    contours_areas = [contourArea(contour) for contour in contours]

    # getting current image col lists
    image_names = [image_name for _ in contours_num_range]
    contour_types = [contour_type for _ in contours_num_range]

    # assembling contours dict
    contours_dict = {'image_name': image_names,
                     'contour': contours,
                     'contour_type': contour_types,
                     'coords': contours_coords,
                     'area': contours_areas}

    # assembling contours df
    contours_df = DataFrame(contours_dict)

    # returning contours df
    return contours_df


def get_autophagy_df(cell_masks_folder: str,
                     foci_masks_folder: str,
                     cell_min_area: int,
                     foci_min_area: int
                     ) -> DataFrame:
    """
    Given paths to folders containing
    segmentation masks (cells/foci),
    returns an autophagy analysis data
    frame.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting images input folder
    images_list = get_specific_files_in_folder(path_to_folder=cell_masks_folder,
                                               extension='.tif')

    # iterating over images list
    for image_name in images_list:

        # getting current image paths
        cell_masks_path = join(cell_masks_folder,
                               image_name)
        foci_masks_path = join(foci_masks_folder,
                               image_name)

        # getting current image contours dfs
        cell_contours_df = get_contours_df(image_name=image_name,
                                           image_path=cell_masks_path,
                                           contour_type='cell')
        foci_contours_df = get_contours_df(image_name=image_name,
                                           image_path=foci_masks_path,
                                           contour_type='foci')

        # filtering dfs by respective minimums
        cell_contours_df = cell_contours_df[cell_contours_df['area'] >= cell_min_area]
        foci_contours_df = foci_contours_df[foci_contours_df['area'] >= foci_min_area]

        # appending dfs to dfs list
        dfs_list.append(cell_contours_df)
        dfs_list.append(foci_contours_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def draw_cell_foci_contours(df: DataFrame,
                            images_folder: str,
                            output_folder: str,
                            color_dict: dict
                            ) -> None:
    """
    Given an autophagy analysis df,
    adds cells/foci contours on top
    of original images, saving overlays
    in output folder.
    """
    # defining group cols
    group_cols = 'image_name'

    # grouping df
    df_groups = df.groupby(group_cols)

    # defining starter for current_img_index
    current_img_index = 1

    # getting contours total
    contours_num = len(df_groups)

    # iterating over df groups
    for image_name, df_group in df_groups:

        # printing execution message
        base_string = f'adding overlays to image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_img_index,
                               total=contours_num)

        # getting current image path
        current_image_path = join(images_folder,
                                  image_name)

        # opening current image
        base_img = imread(current_image_path,
                          -1)

        # converting current image to rgb
        base_img = cvtColor(base_img, COLOR_GRAY2BGR)

        # getting colors based on color dict
        cell_color = color_dict['cell']
        foci_color = color_dict['foci']

        # getting current image cell/foci dfs
        cell_df = df_group[df_group['contour_type'] == 'cell']
        foci_df = df_group[df_group['contour_type'] == 'foci']

        # getting current image cell/foci contours
        cell_contours = cell_df['contour'].to_list()
        foci_contours = foci_df['contour'].to_list()

        # drawing contours
        drawContours(base_img, cell_contours, -1, cell_color, 1)
        drawContours(base_img, foci_contours, -1, foci_color, 1)

        # defining current image save path
        current_image_save_path = join(output_folder,
                                       image_name)

        # saving current image
        imwrite(current_image_save_path,
                base_img)

        # updating current_img_index
        current_img_index += 1


def generate_autophagy_df(images_folder: str,
                          cell_masks_folder: str,
                          foci_masks_folder: str,
                          output_folder: str,
                          ) -> None:
    """
    Given paths to folders containing segmentation
    masks (cells/foci), analyses images to generate
    an autophagy analysis data frame.
    """
    # getting autophagy df
    print('getting autophagy df...')
    autophagy_df = get_autophagy_df(cell_masks_folder=cell_masks_folder,
                                    foci_masks_folder=foci_masks_folder,
                                    cell_min_area=CELL_MIN_AREA,
                                    foci_min_area=FOCI_MIN_AREA)

    # saving autophagy df
    print('saving autophagy df...')
    save_name = 'autophagy_df.pickle'
    save_path = join(output_folder,
                     save_name)
    autophagy_df.to_pickle(save_path)

    # drawing contours
    print('adding contours overlays...')
    draw_cell_foci_contours(df=autophagy_df,
                            images_folder=images_folder,
                            output_folder=output_folder,
                            color_dict=COLOR_DICT)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder
    images_folder = args_dict['images_folder']

    # getting cell folder path
    cell_masks_folder = args_dict['cell_masks_folder']

    # getting foci folder path
    foci_masks_folder = args_dict['foci_masks_folder']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_autophagy_df function
    generate_autophagy_df(images_folder=images_folder,
                          cell_masks_folder=cell_masks_folder,
                          foci_masks_folder=foci_masks_folder,
                          output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
