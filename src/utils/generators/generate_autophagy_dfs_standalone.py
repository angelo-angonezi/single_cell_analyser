# generate autophagy dfs standalone module

print('initializing...')  # noqa

# Code destined to generating autophagy
# data frame, based on segmentation masks.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from os import listdir
from sys import stdout
from cv2 import imwrite
from cv2 import putText
from cv2 import cvtColor
from os.path import join
from pandas import concat
from pandas import Series
from numpy import ndarray
from cv2 import contourArea
from cv2 import boundingRect
from cv2 import drawContours
from pandas import DataFrame
from cv2 import findContours
from cv2 import RETR_EXTERNAL
from cv2 import COLOR_GRAY2BGR
from cv2 import pointPolygonTest
from cv2 import CHAIN_APPROX_NONE
from argparse import ArgumentParser
from cv2 import FONT_HERSHEY_SIMPLEX
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

COLOR_DICT = {'cell': (0, 0, 255),  # red
              'foci': (255, 0, 0)}  # blue

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate autophagy dfs module'

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

    # cell min area param
    parser.add_argument('-cm', '--cell-min-area',
                        dest='cell_min_area',
                        required=False,
                        default=100,
                        help='defines minimum area for a cell (in pixels)')

    # foci min area param
    parser.add_argument('-fm', '--foci-min-area',
                        dest='foci_min_area',
                        required=False,
                        default=2,
                        help='defines minimum area for a foci (in pixels)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# auxiliary functions from aux_funcs module


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


def enter_to_continue():
    """
    Waits for user input ("Enter")
    and once press, continues to run code.
    """
    # defining enter_string
    enter_string = f'press "Enter" to continue'

    # waiting for user input
    input(enter_string)


######################################################################
# defining auxiliary functions


def get_contour_centroid(contour: ndarray) -> tuple:
    """
    Given a contour, returns
    center coordinates in a tuple
    of following structure:
    (cx, cy)
    """
    # getting contour coords
    contour_coords = boundingRect(contour)

    # extracting features from coords tuple
    corner_x, corner_y, width, height = contour_coords

    # getting current center points
    cx = corner_x + (width / 2)
    cy = corner_y + (height / 2)

    # converting cx/cy to ints
    cx = int(cx)
    cy = int(cy)

    # assembling coords tuple
    coords_tuple = (cx, cy)

    # returning coords tuple
    return coords_tuple


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

    # getting current contours indices
    contours_indices = [f for f in contours_num_range]

    # getting current contours coords
    contours_coords = [get_contour_centroid(contour) for contour in contours]

    # getting current contours areas
    contours_areas = [contourArea(contour) for contour in contours]

    # getting current image col lists
    image_names = [image_name for _ in contours_num_range]
    contour_types = [contour_type for _ in contours_num_range]

    # assembling contours dict
    contours_dict = {'image_name': image_names,
                     'contour_index': contours_indices,
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

    # getting images num
    images_num = len(images_list)

    # defining starter for current_img_index
    current_img_index = 1

    # iterating over images list
    for image_name in images_list:

        # printing progress message
        base_string = 'getting autophagy data for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_img_index,
                               total=images_num)

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

        # updating current_img_index
        current_img_index += 1

    # concatenating dfs in dfs list
    print('assembling final df...')
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def draw_single_contour(base_img: ndarray,
                        row_data: Series,
                        color_dict: dict
                        ) -> ndarray:
    """
    Given an open image, draws given
    contour in image, coloring it based
    on contour type and color dict,
    returning image with added overlay.
    """
    # getting current row contour
    contour = row_data['contour']

    # getting current row contour index text
    contour_index = row_data['contour_index']

    # getting current row contour type
    contour_type = row_data['contour_type']

    # getting current row contour label
    current_label = f'{contour_index}'

    # getting current row contour coords
    contour_coords = row_data['coords']
    corner_x, corner_y = contour_coords
    coords_tuple = (corner_x, corner_y)

    # getting color based on color dict
    contour_color = color_dict[contour_type]

    # drawing current contour
    drawContours(base_img,
                 [contour],
                 -1,
                 contour_color,
                 1)

    # adding index label
    putText(base_img,
            current_label,
            coords_tuple,
            FONT_HERSHEY_SIMPLEX,
            0.3,
            contour_color,
            1)

    # returning image with contours
    return base_img


def draw_multiple_contours(df: DataFrame,
                           image_path: str,
                           output_path: str,
                           color_dict: dict
                           ) -> None:
    """
    Given an autophagy df for a single
    image, loads image and adds cell/foci
    overlays, coloring it based on color
    dict, saving overlays image in given
    output path.
    """
    # opening current image
    base_img = imread(image_path,
                      -1)

    # converting current image to rgb
    base_img = cvtColor(base_img, COLOR_GRAY2BGR)

    # getting current df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # adding overlay of current contour
        draw_single_contour(base_img=base_img,
                            row_data=row_data,
                            color_dict=color_dict)

    # saving current image
    imwrite(output_path,
            base_img)


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

        # getting current image input/output paths
        current_image_path = join(images_folder,
                                  image_name)
        current_image_save_path = join(output_folder,
                                       image_name)

        # adding overlays to current image
        draw_multiple_contours(df=df_group,
                               image_path=current_image_path,
                               output_path=current_image_save_path,
                               color_dict=color_dict)

        # updating current_img_index
        current_img_index += 1


def get_cell_foci(cell_contour: ndarray,
                  foci_contours: list
                  ) -> list:
    """
    Given a cell contour, and a list of
    possible foci contours, returns a list
    of foci which are inside given cell.
    """
    # defining placeholder value for valid foci list
    valid_foci = []

    # iterating over foci contours
    for foci_contour in foci_contours:

        # getting current foci contour center coords
        current_foci_coords = get_contour_centroid(foci_contour)

        # getting foci_is_inside_cell bool
        foci_is_inside_cell = pointPolygonTest(cell_contour,
                                               current_foci_coords,
                                               measureDist=False)

        # checking whether current foci contour is inside cell contour
        if foci_is_inside_cell > -1:

            # appending current foci contours to valid foci list
            valid_foci.append(foci_contour)

    # returning valid foci list
    return valid_foci


def get_associations_df(cell_df: DataFrame,
                        foci_df: DataFrame
                        ) -> DataFrame:
    """
    Given a set of cells/foci dfs for
    a single image, returns cells-foci
    associations df.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting cell df rows
    cell_df_rows = cell_df.iterrows()

    # getting foci contours
    foci_contours_col = foci_df['contour']
    foci_contours = foci_contours_col.to_list()

    # iterating over cell df rows
    for row_index, row_data in cell_df_rows:

        # getting current cell info
        current_cell_image_name = row_data['image_name']
        current_cell_id = row_data['contour_index']
        current_cell_contour = row_data['contour']
        current_cell_coords = row_data['coords']
        current_cell_area = row_data['area']

        # getting current foci contours
        current_foci_contours = get_cell_foci(cell_contour=current_cell_contour,
                                              foci_contours=foci_contours)

        # getting current foci count
        current_foci_count = len(current_foci_contours)

        # getting current contours coords
        current_foci_coords = [get_contour_centroid(contour) for contour in current_foci_contours]

        # getting current contours areas
        current_foci_areas = [contourArea(contour) for contour in current_foci_contours]

        # converting areas to Series object
        current_foci_areas_series = Series(current_foci_areas,
                                           dtype=float)

        # getting current contours areas sum/mean/std
        current_foci_areas_sum = current_foci_areas_series.sum()
        current_foci_areas_mean = current_foci_areas_series.mean()
        current_foci_areas_std = current_foci_areas_series.std()

        # assembling current cell dict
        current_cell_dict = {'image_name': current_cell_image_name,
                             'cell_index': current_cell_id,
                             'cell_contour': [current_cell_contour],
                             'cell_coords': [current_cell_coords],
                             'cell_area': current_cell_area,
                             'foci_count': current_foci_count,
                             'foci_contours': [current_foci_contours],
                             'foci_coords': [current_foci_coords],
                             'foci_areas': [current_foci_areas],
                             'foci_areas_sum': current_foci_areas_sum,
                             'foci_areas_mean': current_foci_areas_mean,
                             'foci_areas_std': current_foci_areas_std}

        # assembling current cell df
        current_cell_df = DataFrame(current_cell_dict,
                                    index=[0])

        # appending current df to dfs list
        dfs_list.append(current_cell_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def get_cells_foci_df(df: DataFrame) -> DataFrame:
    """
    Given an autophagy cells/foci df,
    returns a data frame containing
    linked cells-foci contours data.
    """
    # defining group cols
    group_cols = 'image_name'

    # grouping df
    df_groups = df.groupby(group_cols)

    # defining starter for current_img_index
    current_img_index = 1

    # getting contours total
    contours_num = len(df_groups)

    # defining placeholder value for dfs list
    dfs_list = []

    # iterating over df groups
    for image_name, df_group in df_groups:

        # printing execution message
        base_string = f'getting cells-foci associations for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_img_index,
                               total=contours_num)

        # getting current image cell/foci dfs
        cell_df = df_group[df_group['contour_type'] == 'cell']
        foci_df = df_group[df_group['contour_type'] == 'foci']

        # getting current image associations df
        associations_df = get_associations_df(cell_df=cell_df,
                                              foci_df=foci_df)

        # appending current image associations df to dfs list
        dfs_list.append(associations_df)

        # updating current_img_index
        current_img_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list)

    # returning final df
    return final_df


def get_summary_df(df: DataFrame) -> DataFrame:
    """
    Given an autophagy df, returns
    data frame containing summary info.
    """
    # defining placeholder value for dfs list
    dfs_list = []

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
        base_string = f'getting summary info for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_img_index,
                               total=contours_num)

        # getting current image cell/foci dfs
        current_image_cell_df = df_group[df_group['contour_type'] == 'cell']
        current_image_foci_df = df_group[df_group['contour_type'] == 'foci']

        # getting current image cell/foci count
        current_image_cell_count = len(current_image_cell_df)
        current_image_foci_count = len(current_image_foci_df)

        # getting current image cell/foci area
        current_image_cell_area = current_image_cell_df['area'].sum()
        current_image_foci_area = current_image_foci_df['area'].sum()

        # getting current image cell/foci area ratio
        current_area_ratio = current_image_foci_area / current_image_cell_area

        # assembling current image dict
        current_dict = {'image_name': image_name,
                        'cell_count': current_image_cell_count,
                        'total_cell_area': current_image_cell_area,
                        'foci_count': current_image_foci_count,
                        'total_foci_area': current_image_foci_area,
                        'foci_by_cell_area_ratio': current_area_ratio}

        # assembling current image df
        current_df = DataFrame(current_dict,
                               index=[0])

        # appending current df to dfs list
        dfs_list.append(current_df)

        # updating current_img_index
        current_img_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning summary df
    return final_df


def generate_autophagy_dfs(images_folder: str,
                           cell_masks_folder: str,
                           foci_masks_folder: str,
                           output_folder: str,
                           cell_min_area: int,
                           foci_min_area: int
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
                                    cell_min_area=cell_min_area,
                                    foci_min_area=foci_min_area)

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

    # getting cells foci df
    print('establishing cells-foci associations...')
    cells_foci_df = get_cells_foci_df(df=autophagy_df)

    # saving cells-foci df
    print('saving cells-foci df...')
    save_name = 'cells_foci_df.pickle'
    save_path = join(output_folder,
                     save_name)
    cells_foci_df.to_pickle(save_path)

    # dropping unrequired cols
    print('creating analysis df...')
    cols_to_keep = ['image_name',
                    'cell_index',
                    'cell_area',
                    'foci_count',
                    'foci_areas_sum',
                    'foci_areas_mean',
                    'foci_areas_std']
    analysis_df = cells_foci_df[cols_to_keep]

    # saving analysis df
    print('saving analysis df...')
    save_name = 'analysis_df.csv'
    save_path = join(output_folder,
                     save_name)
    analysis_df.to_csv(save_path,
                       index=False)

    # getting summary df
    summary_df = get_summary_df(df=autophagy_df)

    # saving summary df
    print('saving analysis df...')
    save_name = 'summary_df.csv'
    save_path = join(output_folder,
                     save_name)
    summary_df.to_csv(save_path,
                      index=False)

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

    # getting cell min area
    cell_min_area = args_dict['cell_min_area']
    cell_min_area = int(cell_min_area)

    # getting foci min area
    foci_min_area = args_dict['foci_min_area']
    foci_min_area = int(foci_min_area)

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_autophagy_dfs function
    generate_autophagy_dfs(images_folder=images_folder,
                           cell_masks_folder=cell_masks_folder,
                           foci_masks_folder=foci_masks_folder,
                           output_folder=output_folder,
                           cell_min_area=cell_min_area,
                           foci_min_area=foci_min_area)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
