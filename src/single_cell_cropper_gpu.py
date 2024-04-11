# single cell cropper (ml) module

print('initializing...')  # noqa

# Code destined to cropping single cells
# using ML output.

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
from os import environ
from cv2 import imwrite
from os.path import join
from cupy import ndarray
from cupy import asnumpy
from cupy import asarray
from pandas import concat
from os.path import exists
from pandas import read_csv
from pandas import DataFrame
from cupy import pad as np_pad
from argparse import ArgumentParser
from cv2 import resize as cv_resize
from src.utils.aux_funcs import get_etc
from src.utils.aux_funcs import IMAGE_SIZE
from src.utils.aux_funcs import load_bgr_img
from src.utils.aux_funcs import get_time_str
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import print_gpu_usage
from src.utils.aux_funcs import get_current_time
from src.utils.aux_funcs import get_time_elapsed
from src.utils.aux_funcs import get_obbs_from_df
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import add_treatment_col
from src.utils.aux_funcs import get_treatment_dict
from src.utils.aux_funcs import add_experiment_cols
from cupyx.scipy.ndimage import rotate as scp_rotate
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

RESIZE_DIMENSIONS = (100, 100)

# progress bar related
ITERATIONS_TOTAL = 0
CURRENT_ITERATION = 1
IMAGES_TOTAL = 0
CURRENT_IMAGE = 1
CROPS_TOTAL = 0
CURRENT_CROP = 1
START_TIME = 0

# setting tensorflow warnings off
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'single cell cropper - tool used to segment cells based on\n'
    description += 'machine learning output data.'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser
    parser.add_argument('-i', '--images-input-folder',
                        dest='images_input_folder',
                        help='defines path to folder containing images',
                        required=True)

    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folder')

    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        help='defines path to output folder which will contain crops\nand crops info file',
                        required=True)

    parser.add_argument('-d', '--detections-dataframe',
                        dest='detections_df_path',
                        help='defines path to file containing detections info',
                        required=True)

    parser.add_argument('-t', '--detection-threshold',
                        dest='detection_threshold',
                        help='defines detection threshold for ml model results',
                        required=False,
                        default=0.5)

    parser.add_argument('-tr', '--treatment-file',
                        dest='treatment_file',
                        help='defines path to file containing treatment info',
                        required=False,
                        default=None)

    parser.add_argument('-er', '--expansion-ratio',
                        dest='expansion_ratio',
                        help='defines ratio of expansion of width/height to generate larger-than-orig-nucleus crops',
                        required=False,
                        default=1.0)

    parser.add_argument('-f', '--fixed-size',
                        dest='fixed_size_toggle',
                        action='store_true',
                        help='ignores crops original dimensions and crops them according to IMAGE_SIZE in aux_funcs.py',
                        required=False,
                        default=False)

    parser.add_argument('-r', '--resize',
                        dest='resize_toggle',
                        action='store_true',
                        help='resizes all crops to same dimensions',
                        required=False,
                        default=False)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def print_global_progress():
    """
    Takes global parameters and prints
    progress message on console.
    """
    # getting global variables
    global CURRENT_ITERATION
    global ITERATIONS_TOTAL
    global IMAGES_TOTAL
    global CURRENT_IMAGE
    global CROPS_TOTAL
    global CURRENT_CROP
    global START_TIME

    # getting current progress
    progress_ratio = CURRENT_ITERATION / ITERATIONS_TOTAL
    progress_percentage = progress_ratio * 100

    # getting current time
    current_time = get_current_time()

    # getting time elapsed
    time_elapsed = get_time_elapsed(start_time=START_TIME,
                                    current_time=current_time)

    # getting estimated time of completion
    etc = get_etc(time_elapsed=time_elapsed,
                  current_iteration=CURRENT_ITERATION,
                  iterations_total=ITERATIONS_TOTAL)

    # converting times to adequate format
    time_elapsed_str = get_time_str(time_in_seconds=time_elapsed)
    etc_str = get_time_str(time_in_seconds=etc)

    # defining progress string
    progress_string = f'generating crops for image {CURRENT_IMAGE} of {IMAGES_TOTAL}... '
    # progress_string += f'| crop: {CURRENT_CROP}/{CROPS_TOTAL} '
    progress_string += f'| crop: {CURRENT_ITERATION}/{ITERATIONS_TOTAL} '
    progress_string += f'| progress: {progress_percentage:02.2f}% '
    progress_string += f'| time elapsed: {time_elapsed_str} '
    progress_string += f'| ETC: {etc_str}'
    progress_string += '   '

    # printing execution message
    flush_or_print(string=progress_string,
                   index=CURRENT_ITERATION,
                   total=ITERATIONS_TOTAL)


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
    resized_image = cv_resize(crop, resize_dimensions)

    # returning resized image
    return resized_image


def black_pixels_in_crop(crop: ndarray) -> bool:
    """
    Given an array, returns True if array
    contains black pixels (pixel_value=0),
    or False otherwise.
    :param crop: Array. Represents an image crop.
    :return: Boolean. Represents whether a crop
    contains black pixels (meaning it is in image
    corners).
    """
    # checking if there are black pixels in array
    black_pixels_in_array = 0 in crop

    # returning boolean
    return black_pixels_in_array


def get_crop_coordinates(cx: int,
                         cy: int,
                         width: float,
                         height: float
                         ) -> tuple:
    """
    Given a set of coords and dims,
    returns crop coordinates:
    (left, right, top, bottom)
    """
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

    # assembling coords tuple
    coords_tuple = (left, right, top, bottom)

    # returning coords tuple
    return coords_tuple


def rotate_image(image: ndarray,
                 angle: float,
                 pivot: tuple,
                 roi_box_width: float
                 ) -> ndarray:
    """
    Given an image, and angle and coordinates
    to pivot, returns rotated image.
    """
    # defining image pads
    pad_x = [image.shape[1] - pivot[0], pivot[0]]
    pad_y = [image.shape[0] - pivot[1], pivot[1]]

    # padding image
    padded_image = np_pad(array=image,
                          pad_width=[pad_y,    # tuple defining above-below padding
                                     pad_x,    # tuple defining left-right padding
                                     [0, 0]],  # tuple defining z-dim padding
                          mode='constant')     # fills background with zeroes

    # getting ROI crop coords
    padded_image_shape = padded_image.shape
    cx = padded_image_shape[1] / 2
    cy = padded_image_shape[0] / 2
    cx = int(cx)
    cy = int(cy)
    crop_coords = get_crop_coordinates(cx=cx, cy=cy, width=roi_box_width, height=roi_box_width)
    left, right, top, bottom = crop_coords

    # cropping ROI (optimized code since rotation is costly)
    # cropped_image = padded_image[left:right, top:bottom]
    cropped_image = padded_image[top:bottom, left:right]

    # rotating image
    rotated_image = scp_rotate(cropped_image, angle, reshape=False)

    # returning rotated image
    return rotated_image


def crop_single_obb(image: ndarray,
                    obb: tuple,
                    fixed_size_toggle: bool,
                    expansion_ratio: float = 1.0
                    ) -> ndarray:
    """
    Given an array representing an image,
    and a tuple containing obb's info,
    returns given obb crop, rotated to
    be aligned to x-axis.
    """
    # getting current obb info
    cx, cy, width, height, angle, cell_class = obb

    # expanding with/height
    width = width * expansion_ratio
    height = height * expansion_ratio

    # checking fixed size toggle
    if fixed_size_toggle:

        # redefining width/height based on aux_funcs param
        width, height = IMAGE_SIZE

    # getting major axis
    major_axis = width if width > height else height
    major_axis = major_axis * 2

    # getting rotation angle (opposite to OBB angle, since the image
    # will be rotated to match OBB orientation)
    rotation_angle = angle * (-1)

    # rotating current image to match current obb angle
    rotated_image = rotate_image(image=image,
                                 angle=rotation_angle,
                                 pivot=(cx, cy),
                                 roi_box_width=major_axis)

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

    # returning crop
    return image_crop


def crop_multiple_obbs(image: ndarray,
                       image_name: str,
                       obbs_list: list,
                       output_folder: str,
                       expansion_ratio: float,
                       fixed_size_toggle: bool,
                       resize_toggle: bool
                       ) -> DataFrame:
    """
    Given an array representing an image,
    and a list of tuples containing obb's
    info, crops obbs in current image,
    saving cropped images and info file
    in given output folder, returning
    a data frame with crops info.
    """
    # getting global parameters
    global ITERATIONS_TOTAL, CURRENT_ITERATION

    # getting obbs total
    obbs_total = len(obbs_list)
    obbs_total_str = str(obbs_total)
    obbs_total_str_len = len(obbs_total_str)

    # updating global parameters
    global CROPS_TOTAL, CURRENT_CROP
    CROPS_TOTAL = obbs_total
    CURRENT_CROP = 1

    # defining placeholder value for dfs list
    dfs_list = []

    # getting current image min/max
    img_min = image.min()
    img_max = image.max()

    # iterating over obbs in obbs list
    for obb_index, obb in enumerate(obbs_list, 1):

        # getting current crop string
        current_crop_str = f'{obb_index:0{obbs_total_str_len}d}'

        # printing execution message
        print_global_progress()

        # getting current crop output name/path
        current_crop_output_name = f'{image_name}_'
        current_crop_output_name += f'crop_{current_crop_str}'
        current_crop_output_name_w_extension = f'{current_crop_output_name}.tif'
        current_crop_output_path = join(output_folder,
                                        current_crop_output_name_w_extension)

        # getting current obb info
        cx, cy, width, height, angle, cell_class = obb

        # assembling current crop dict
        current_crop_dict = {'img_name': image_name,
                             'img_min': img_min,
                             'img_max': img_max,
                             'crop_index': current_crop_str,
                             'crop_name': current_crop_output_name,
                             'cx': cx,
                             'cy': cy,
                             'width': width,
                             'height': height,
                             'angle': angle,
                             'class': cell_class}

        # getting current crop df
        current_crop_df = DataFrame(current_crop_dict,
                                    index=[0])

        # appending current crop df to dfs list
        dfs_list.append(current_crop_df)

        # updating global parameters
        CURRENT_CROP += 1
        CURRENT_ITERATION += 1

        # checking if current crop already exists
        if exists(current_crop_output_path):

            # skipping current crop image generation
            continue

        # getting current obb crop
        current_obb_crop = crop_single_obb(image=image,
                                           obb=obb,
                                           fixed_size_toggle=fixed_size_toggle,
                                           expansion_ratio=expansion_ratio)

        # checking resize toggle
        if resize_toggle:

            # getting global parameters
            global RESIZE_DIMENSIONS

            # resizing image to specified dimensions
            current_obb_crop = resize_crop(crop=current_obb_crop,
                                           resize_dimensions=RESIZE_DIMENSIONS)

        # converting current crop back to numpy (necessary to save with cv2.imwrite function)
        current_obb_crop = asnumpy(current_obb_crop)

        # saving current crop
        imwrite(filename=current_crop_output_path,
                img=current_obb_crop)

    # concatenating dfs in dfs list
    crops_df = concat(dfs_list,
                      ignore_index=True)

    # returning crops info df
    return crops_df


def get_single_image_crops(image: ndarray,
                           image_name: str,
                           image_group: DataFrame,
                           output_folder: str,
                           expansion_ratio: float,
                           fixed_size_toggle: bool,
                           resize_toggle: bool
                           ) -> DataFrame:
    """
    Given an array representing an image,
    and a data frame representing current
    image obbs detections, saves crops
    of obbs in given output folder, returning
    a data frame with crops info.
    """
    # sorting df by cx (ensures that different codes follow the same order)
    image_group = image_group.sort_values(by='cx')

    # getting current image obbs
    current_image_obbs = get_obbs_from_df(df=image_group)

    # cropping obbs in current image group
    crops_df = crop_multiple_obbs(image=image,
                                  image_name=image_name,
                                  obbs_list=current_image_obbs,
                                  output_folder=output_folder,
                                  expansion_ratio=expansion_ratio,
                                  fixed_size_toggle=fixed_size_toggle,
                                  resize_toggle=resize_toggle)

    # returning crops df
    return crops_df


def get_multiple_image_crops(consolidated_df: DataFrame,
                             input_folder: str,
                             images_extension: str,
                             output_folder: str,
                             expansion_ratio: float,
                             fixed_size_toggle: bool,
                             resize_toggle: bool
                             ) -> DataFrame:
    """
    Given a detections consolidated data frame,
    a path to an input folder containing images,
    saves obbs crops in output folder, returning
    a data frame with crops info.
    :param consolidated_df: DataFrame. Represents obbs
    detections for images in input folder (in model output format).
    :param input_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param output_folder: String. Represents a path to a folder.
    :param expansion_ratio: Float. Represents a ratio to expand width/height.
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
    f_string = f'a total of {ITERATIONS_TOTAL} obbs were found in detections file.'
    print(f_string)

    # grouping detections data frame by images
    image_groups = consolidated_df.groupby('img_file_name')

    # getting total number of images
    image_total = len(image_groups)

    # updating global parameters
    global IMAGES_TOTAL
    IMAGES_TOTAL = image_total

    # defining placeholder value for dfs list
    dfs_list = []

    # iterating over images groups
    for image_name, image_group in image_groups:

        # getting image name string
        image_name = str(image_name)

        # getting current image name with extension
        image_name_w_extension = f'{image_name}{images_extension}'

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
        current_image_array = load_bgr_img(image_path=current_image_path)

        # converting image to cupy array
        current_image_array = asarray(current_image_array)

        # running single image cropper
        crops_df = get_single_image_crops(image=current_image_array,
                                          image_name=image_name,
                                          image_group=image_group,
                                          output_folder=output_folder,
                                          expansion_ratio=expansion_ratio,
                                          fixed_size_toggle=fixed_size_toggle,
                                          resize_toggle=resize_toggle)

        # appending current image crops df to dfs list
        dfs_list.append(crops_df)

        # updating global parameters
        global CURRENT_IMAGE
        CURRENT_IMAGE += 1

    # printing execution message
    f_string = f'all crops generated!'
    print(f_string)

    # concatenating dfs in dfs list
    crops_df = concat(dfs_list,
                      ignore_index=True)

    # returning crops info df
    return crops_df


def single_cell_cropper(input_folder: str,
                        images_extension: str,
                        detections_df_path: str,
                        detection_threshold: float,
                        treatment_file: str or None,
                        expansion_ratio: float,
                        output_folder: str,
                        fixed_size_toggle: bool,
                        resize_toggle: bool
                        ) -> None:
    """
    Given execution parameters, runs
    cropping function on multiple images,
    saving output crops and crops info file.
    """
    # getting data from consolidated df csv
    print('getting data from consolidated df...')
    consolidated_df = read_csv(detections_df_path)

    # filtering df
    print('filtering df by detection threshold...')
    filtered_df = consolidated_df.loc[consolidated_df['detection_threshold'] >= detection_threshold]

    # sorting df
    print('sorting data frame...')
    sorted_df = filtered_df.sort_values('img_file_name')

    # getting current time
    current_time = get_current_time()

    # updating global variables
    global START_TIME
    START_TIME = current_time

    # running multiple image cropper function
    print('initializing crops generator...')
    crops_df = get_multiple_image_crops(consolidated_df=sorted_df,
                                        input_folder=input_folder,
                                        images_extension=images_extension,
                                        output_folder=output_folder,
                                        expansion_ratio=expansion_ratio,
                                        fixed_size_toggle=fixed_size_toggle,
                                        resize_toggle=resize_toggle)
    print(f'image crops saved at "{output_folder}".')

    # adding cols to crops df
    print('adding analysis cols to crops df...')

    # adding experiment cols
    add_experiment_cols(df=crops_df,
                        file_name_col='img_name')

    # checking if treatment file is not None
    if treatment_file is not None:

        # getting treatment dict
        treatment_dict = get_treatment_dict(treatment_file=treatment_file)

        # adding treatment col
        add_treatment_col(df=crops_df,
                          treatment_dict=treatment_dict)

    # saving final crops df in output folder
    print('saving crops info df...')
    save_name = 'crops_info.csv'
    save_path = join(output_folder,
                     save_name)
    # crops_df.to_csv(save_path,
    #                 index=False)
    print(f'crops info df saved at "{save_path}".')

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

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting detections df path
    detections_df_path = args_dict['detections_df_path']

    # getting detection threshold
    detection_threshold = args_dict['detection_threshold']
    detection_threshold = float(detection_threshold)

    # getting treatment file path
    treatment_file = args_dict['treatment_file']

    # getting expansion ratio
    expansion_ratio = args_dict['expansion_ratio']
    expansion_ratio = float(expansion_ratio)

    # getting fixed size toggle
    fixed_size_toggle = args_dict['fixed_size_toggle']

    # getting resize toggle
    resize_toggle = args_dict['resize_toggle']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    print_gpu_usage()

    # waiting for user input
    enter_to_continue()

    # running single cell cropper function
    single_cell_cropper(input_folder=input_folder,
                        images_extension=images_extension,
                        detections_df_path=detections_df_path,
                        detection_threshold=detection_threshold,
                        treatment_file=treatment_file,
                        expansion_ratio=expansion_ratio,
                        output_folder=output_folder,
                        fixed_size_toggle=fixed_size_toggle,
                        resize_toggle=resize_toggle)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
