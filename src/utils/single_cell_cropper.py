# single cell cropper module

print('initializing...')  # noqa

# code destined to cropping single cells
# using ML output.

# debug execution
# python -m src.utils.single_cell_cropper -i .\\data\\input_imgs\\ -d .\\data\\ml_detections\\r2cnn_detections.csv -o .\\data\\output_crops\\

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
from time import sleep
from cv2 import imread
from cv2 import imwrite
from os.path import join
from numpy import ndarray
from pandas import concat
from os.path import exists
from pandas import DataFrame
from numpy import pad as np_pad
from argparse import ArgumentParser
from src.utils.aux_funcs import flush_or_print
from scipy.ndimage import rotate as scp_rotate
from src.utils.aux_funcs import get_obbs_from_df
from src.utils.aux_funcs import get_data_from_consolidated_df
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

#####################################################################
# defining global variables

ITERATIONS_TOTAL = 0
CURRENT_ITERATION = 1
RESIZE_DIMENSIONS = (500, 500)

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "single cell cropper - tool used to segment cells based on\n"
    description += "machine learning output data.\n"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser
    parser.add_argument('-i', '--images-input-folder',
                        dest='images_input_folder',
                        help='defines path to folder containing images',
                        required=True)

    parser.add_argument('-d', '--detections-dataframe',
                        dest='detections_df_path',
                        help='defines path to file containing detections info',
                        required=True)

    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        help='defines path to output folder which will contain crops\nand crops info file',
                        required=True)

    parser.add_argument('-r', '--resize',
                        dest='resize_toggle',
                        action='store_true',
                        help='resizes all crops to same dimensions',
                        default=False,
                        required=False)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def resize_crop(crop: ndarray,
                resize_dimensions: tuple
                ) -> ndarray:
    """
    Given a crop and resize dimensions,
    returns resized crop.
    :param crop: Array. Represents an image crop.
    :param resize_dimensions: Tuple, represents final
    crop desired dimensions.
    :return: Array. Represents resized image crop.
    """
    # getting resized image
    # TODO: finish this function altering this line.
    resized_image = crop

    # returning resized image
    return resized_image


def rotate_image(image: ndarray,
                 angle: float,
                 pivot: tuple
                 ) -> ndarray:
    """
    Given an image, and angle and coordinates
    to pivot, returns rotated image.
    :param image: Array. Represents an open image.
    :param angle: Float. Represents rotation angle.
    :param pivot: Float. Represents pivot coordinates (cx, cy).
    :return: Array. Represents rotated image.
    """
    # defining image pads
    pad_x = [image.shape[1] - pivot[0], pivot[0]]
    pad_y = [image.shape[0] - pivot[1], pivot[1]]

    # padding image
    padded_image = np_pad(image, [pad_y, pad_x, [0, 0]], 'constant')

    # rotating image
    rotated_image = scp_rotate(padded_image, angle, reshape=False)

    # returning rotated image
    return rotated_image


def crop_single_obb(image: ndarray,
                    obb: tuple,
                    resize_toggle: bool
                    ) -> ndarray:
    """
    Given an array representing an image,
    and a tuple containing obb's info,
    returns given obb crop, rotated to
    be aligned to x-axis.
    :param image: Array. Represents an open image.
    :param obb: Tuple. Represents obb's info.
    :param resize_toggle: Boolean. Represents a toggle.
    :return: Array. Represents obb crop from image.
    """
    # getting current obb info
    cx, cy, width, height, angle = obb

    # rotating current image to match current obb angle
    rotated_image = rotate_image(image=image,
                                 angle=angle,
                                 pivot=(cx, cy))

    # getting new cx, cy values
    rotated_image_shape = rotated_image.shape
    cx = rotated_image_shape[0] / 2
    cy = rotated_image_shape[1] / 2

    # getting margins
    top = cy - (height / 2)
    bottom = cy + (height / 2)
    left = cx - (width / 2)
    right = cx + (width / 2)

    # converting margins to integers
    top = int(top)
    bottom = int(bottom)
    left = int(left)
    right = int(right)

    # cropping image (using numpy slicing)
    image_crop = rotated_image[left:right, top:bottom]

    # TODO: add logic to deal with black borders

    # checking resize toggle
    if resize_toggle:

        # getting global parameters
        global RESIZE_DIMENSIONS

        # resizing image to specified dimensions
        image_crop = resize_crop(crop=image_crop,
                                 resize_dimensions=RESIZE_DIMENSIONS)

    # returning crop
    return image_crop


def crop_multiple_obbs(image: ndarray,
                       image_name: str,
                       obbs_list: list,
                       output_folder: str,
                       resize_toggle: bool,
                       progress_string: str
                       ) -> DataFrame:
    """
    Given an array representing an image,
    and a list of tuples containing obb's
    info, crops obbs in current image,
    saving cropped images and info file
    in given output folder, returning
    a data frame with crops info.
    :param image: Array. Represents an open image.
    :param image_name: String. Represents an image name.
    :param obbs_list: List. Represents a list of obbs.
    :param output_folder: String. Represents a path to a folder.
    :param resize_toggle: Boolean. Represents a toggle.
    :param progress_string: String. Represents a progress string.
    :return: Data Frame. Represents crops info.
    """
    # getting global parameters
    global ITERATIONS_TOTAL, CURRENT_ITERATION

    # getting obbs total
    obbs_total = len(obbs_list)
    obbs_total_str = str(obbs_total)
    obbs_total_str_len = len(obbs_total_str)

    # defining placeholder value for dfs list
    dfs_list = []

    # iterating over obbs in obbs list
    for obb_index, obb in enumerate(obbs_list, 1):

        # getting current crop string
        current_crop_str = f'{obb_index:0{obbs_total_str_len}d}'

        # getting current percentage progress
        current_percentage_ratio = CURRENT_ITERATION / ITERATIONS_TOTAL
        current_percentage_progress = current_percentage_ratio * 100
        current_percentage_round_progress = round(current_percentage_progress, 2)
        current_percentage_round_progress_str = '{:.2f}'.format(current_percentage_round_progress)

        # printing execution message
        current_progress_string = f'{progress_string} '
        current_progress_string += f'(crop: {current_crop_str} '
        current_progress_string += f'of {obbs_total}) '
        current_progress_string += f'| {current_percentage_round_progress_str}%'
        flush_or_print(string=current_progress_string,
                       index=CURRENT_ITERATION,
                       total=ITERATIONS_TOTAL)

        # updating global parameters
        CURRENT_ITERATION += 1

        # getting current obb crop
        current_obb_crop = crop_single_obb(image=image,
                                           obb=obb,
                                           resize_toggle=resize_toggle)

        # getting current crop output name/path
        current_crop_output_name = f'{image_name}_'
        current_crop_output_name += f'crop_{current_crop_str}'
        current_crop_output_name_w_extension = f'{current_crop_output_name}.tif'
        current_crop_output_path = join(output_folder,
                                        current_crop_output_name_w_extension)

        # saving current crop
        imwrite(filename=current_crop_output_path,
                img=current_obb_crop)

        # getting current obb info
        cx, cy, width, height, angle = obb

        # assembling current crop dict
        current_crop_dict = {'img_name': image_name,
                             'crop_index': current_crop_str,
                             'crop_name': current_crop_output_name,
                             'cx': cx,
                             'cy': cy,
                             'width': width,
                             'height': height,
                             'angle': angle}

        # getting current crop df
        current_crop_dict = DataFrame(current_crop_dict,
                                      index=[0])

        # appending current crop df to dfs list
        dfs_list.append(current_crop_dict)

    # concatenating dfs in dfs list
    crops_df = concat(dfs_list,
                      ignore_index=True)

    # returning crops info df
    return crops_df


def get_single_image_crops(image: ndarray,
                           image_name: str,
                           image_group: DataFrame,
                           output_folder: str,
                           resize_toggle: bool,
                           progress_string: str
                           ) -> DataFrame:
    """
    Given an array representing an image,
    and a data frame representing current
    image obbs detections, saves crops
    of obbs in given output folder, returning
    a data frame with crops info.
    :param image: Array. Represents an open image.
    :param image_name: String. Represents an image name.
    :param image_group: DataFrame. Represents current image obbs.
    :param output_folder: String. Represents a path to a folder.
    :param resize_toggle: Boolean. Represents a toggle.
    :param progress_string: String. Represents a progress string.
    :return: Data Frame. Represents crops info.
    """
    # getting current image obbs
    current_image_obbs = get_obbs_from_df(df=image_group)

    # cropping obbs in current image group
    crops_df = crop_multiple_obbs(image=image,
                                  image_name=image_name,
                                  obbs_list=current_image_obbs,
                                  output_folder=output_folder,
                                  resize_toggle=resize_toggle,
                                  progress_string=progress_string)

    # returning crops df
    return crops_df


def get_multiple_image_crops(consolidated_df: DataFrame,
                             input_folder: str,
                             output_folder: str,
                             resize_toggle: bool
                             ) -> DataFrame:
    """
    Given ML detections consolidated data frame,
    a path to an input folder containing images,
    saves obbs crops in output folder, returning
    a data frame with crops info.
    :param consolidated_df: DataFrame. Represents ML obbs
    detections for images in input folder.
    :param input_folder: String. Represents a path to a folder.
    :param output_folder: String. Represents a path to a folder.
    :param resize_toggle: Boolean. Represents a toggle.
    :return: Data Frame. Represents crops info.
    """
    # getting global parameters
    global ITERATIONS_TOTAL

    # getting number of iterations
    iterations_total = len(consolidated_df)

    # updating global parameters
    ITERATIONS_TOTAL = iterations_total

    # printing execution message
    f_string = f'a total of {ITERATIONS_TOTAL} obbs were found in ml detections file.'
    print(f_string)

    # grouping detections data frame by images
    image_groups = consolidated_df.groupby('img_file_name')

    # getting total number of images
    image_total = len(image_groups)
    image_total_str = str(image_total)
    image_total_str_len = len(image_total_str)

    # defining placeholder value for dfs list
    dfs_list = []

    # iterating over images groups
    for image_index, (image_name, image_group) in enumerate(image_groups, 1):

        # getting current image name with extension
        image_name_w_extension = f'{image_name}.tif'

        # getting current image path in input folder
        current_image_path = join(input_folder, image_name_w_extension)

        # checking if current image is in input folder
        image_is_in_folder = exists(current_image_path)

        # if file is not in folder
        if not image_is_in_folder:

            # printing error message
            e_string = f'Unable to find image "{image_name_w_extension}" in given input folder.\n'
            e_string += f'Skipping to next image.'
            print(e_string)

            # skipping to next image
            continue

        # reading current image with cv2
        current_image_array = imread(current_image_path)

        # assembling current progress string
        progress_string = f'generating crops for image {image_index:0{image_total_str_len}d}'
        progress_string += f' of {image_total}...'

        # running single image cropper
        crops_df = get_single_image_crops(image=current_image_array,
                                          image_name=image_name,
                                          image_group=image_group,
                                          output_folder=output_folder,
                                          resize_toggle=resize_toggle,
                                          progress_string=progress_string)

        # appending current image crops df to dfs list
        dfs_list.append(crops_df)

    # printing execution message
    f_string = f'all crops generated!'
    print(f_string)

    # concatenating dfs in dfs list
    crops_df = concat(dfs_list,
                      ignore_index=True)

    # returning crops info df
    return crops_df


def single_cell_cropper(input_folder: str,
                        detections_df_path: str,
                        output_folder: str,
                        resize_toggle: bool
                        ) -> None:
    """
    Given execution parameters, runs
    cropping function on multiple images.
    :param input_folder: String. Represents a path to a folder.
    :param detections_df_path: String. Represents a path to a file.
    :param output_folder: String. Represents a path to a folder.
    :param resize_toggle: Bool. Represents a toggle.
    :return: None.
    """
    # getting data from consolidated df csv
    print('getting data from consolidated df...')
    consolidated_df = get_data_from_consolidated_df(consolidated_df_file_path=detections_df_path)

    # sorting df
    print('sorting data frame...')
    sorted_df = consolidated_df.sort_values('img_file_name')

    # running multiple image cropper function
    print('initializing crops generator...')
    crops_df = get_multiple_image_crops(consolidated_df=sorted_df,
                                        input_folder=input_folder,
                                        output_folder=output_folder,
                                        resize_toggle=resize_toggle)
    print(f'image crops saved at "{output_folder}".')

    # saving final crops df in output folder
    print('saving crops info df...')
    df_name = 'crops_info'
    df_name_w_extension = f'{df_name}.csv'
    df_path = join(output_folder,
                   df_name_w_extension)
    crops_df.to_csv(df_path,
                    index=False)
    print(f'crops info df saved at "{df_path}".')

######################################################################
# defining main function


def main():
    """
    Runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['images_input_folder']

    # getting detections df path
    detections_df_path = args_dict['detections_df_path']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting resize toggle
    resize_toggle = args_dict['resize_toggle']

    # running single cell cropper function
    single_cell_cropper(input_folder=input_folder,
                        detections_df_path=detections_df_path,
                        output_folder=output_folder,
                        resize_toggle=resize_toggle)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
