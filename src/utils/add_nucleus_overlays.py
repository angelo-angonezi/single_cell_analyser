# add nucleus overlay module

print('initializing...')  # noqa

# Code destined to generating images with nucleus
# overlays based on model detections (and GT annotations).

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from numpy import intp
from cv2 import imread
from cv2 import imwrite
from cv2 import putText
from cv2 import cvtColor
from os.path import join
from numpy import ndarray
from pandas import concat
from pandas import Series
from cv2 import boxPoints
from pandas import read_csv
from pandas import DataFrame
from cv2 import drawContours
from cv2 import COLOR_BGR2RGB
from cv2 import COLOR_RGB2BGR
from argparse import ArgumentParser
from cv2 import FONT_HERSHEY_SIMPLEX
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

COLOR_DICT = {'model': (0, 102, 204),
              'fornma': (0, 204, 102),
              'DT': (255, 153, 102)}

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "add nucleus overlays to images based on model detections (and GT annotations)"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    input_help = 'defines input folder (folder containing images)'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=input_help)

    # image extension param
    extension_help = 'defines extension (.tif, .png, .jpg) of images in input folder'
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help=extension_help)

    # detection file param
    detection_help = 'defines path to csv file containing model detections'
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help=detection_help)

    # gt file param
    gt_help = 'defines path to csv file containing ground-truth annotations'
    gt_help += '(if none is passed, adds only model detections)'
    parser.add_argument('-g', '--ground-truth-file',
                        dest='ground_truth_file',
                        required=False,
                        help=gt_help)

    # output folder param
    output_help = 'defines output folder (folder that will contain outlined images)'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help=output_help)

    # detection threshold param
    threshold_help = 'defines threshold to be applied (filters detections OBBs based on detection threshold)'
    parser.add_argument('-t', '--detection-threshold',
                        dest='detection_threshold',
                        required=False,
                        default=0.6,
                        help=threshold_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_merged_detection_annotation_df(detections_df_path: str,
                                       annotations_df_path: str or None
                                       ) -> DataFrame:
    """
    Given a path to detections df and annotations df,
    returns merged df, containing new column "evaluator",
    representing detection/annotation info.
    :param detections_df_path: String. Represents a path to a file.
    :param annotations_df_path: String. Represents a path to a file.
    :return: DataFrame. Represents merged detection/annotation data.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # reading detections file
    print('reading detections file...')
    detections_df = read_csv(detections_df_path)

    # adding evaluator constant column
    detections_df['evaluator'] = 'model'

    # adding detections df to dfs_list
    dfs_list.append(detections_df)

    # checking ground_truth_file_path
    if annotations_df_path is not None:

        # reading gt file
        print('reading ground-truth file...')
        ground_truth_df = read_csv(annotations_df_path)

        # adding evaluator constant column
        ground_truth_df['evaluator'] = 'fornma'

        # adding annotations df to dfs_list
        dfs_list.append(ground_truth_df)

    # concatenating dfs in dfs_list
    merged_df = concat(dfs_list)

    # returning merged df
    return merged_df


def draw_rectangle(open_img: ndarray,
                   cx: float,
                   cy: float,
                   width: float,
                   height: float,
                   angle: float,
                   color: tuple
                   ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with OBB overlay.
    """
    # get the corner points
    box = boxPoints(((cx, cy),
                     (width, height),
                     angle))

    # converting corners format
    box = intp(box)

    # drawing lines
    drawContours(open_img,
                 [box],
                 -1,
                 color,
                 2)

    # returning modified image
    return open_img


def add_single_overlay(open_img: ndarray,
                       obbs_df_row: Series,
                       color_dict: dict
                       ) -> None:
    """
    Given an open image and respective obb row
    (extracted from merged_df), returns image
    with given obb overlay.
    :param open_img: ndarray. Represents an open image.
    :param obbs_df_row: Series. Represents single obb data.
    :param color_dict: Dictionary. Represents colors to be used in overlays.
    :return: None.
    """
    # getting current row bounding box info
    cx = float(obbs_df_row['cx'])
    cy = float(obbs_df_row['cy'])
    width = float(obbs_df_row['width'])
    height = float(obbs_df_row['height'])
    angle = float(obbs_df_row['angle'])
    det_class = str(obbs_df_row['class'])
    evaluator = str(obbs_df_row['evaluator'])

    # defining color for overlay
    overlay_color = color_dict[evaluator]

    # adding rectangle overlay
    draw_rectangle(open_img=open_img,
                   cx=cx,
                   cy=cy,
                   width=width,
                   height=height,
                   angle=angle,
                   color=overlay_color)

    # adding class text
    putText(open_img,
            det_class,
            (int(cx), int(cy)),
            FONT_HERSHEY_SIMPLEX,
            0.0001,  # TODO: change this once we got more classes and think of better way to display class info
            overlay_color,
            2)


def add_multiple_overlays(open_img: ndarray,
                          current_image_df: DataFrame,
                          color_dict: dict
                          ) -> None:
    """
    Given an open image and current image data frame,
    returns image with detection/annotation overlays.
    :param open_img: ndarray. Represents an open image.
    :param current_image_df: DataFrame. Represents current image detection/annotation data.
    :param color_dict: Dictionary. Represents colors to be used in overlays.
    :return: None.
    """
    # getting df rows
    df_rows = current_image_df.iterrows()

    # iterating over df_rows
    for row_index, row_data in df_rows:

        # adding current row overlay
        add_single_overlay(open_img=open_img,
                           obbs_df_row=row_data,
                           color_dict=color_dict)


def add_overlays_to_single_image(image_name: str,
                                 image_path: str,
                                 merged_df: DataFrame,
                                 detection_threshold: float,
                                 output_path: str,
                                 color_dict: dict
                                 ) -> None:
    """
    Given an image name and path, and a merged
    detections/annotations data frame, save image
    with overlays in given output path.
    :param image_name: String. Represents an image name.
    :param image_path: String. Represents a file path.
    :param merged_df: DataFrame. Represents detections/annotations data.
    :param detection_threshold: Float. Represents detection threshold to be applied as filter.
    :param output_path: String. Represents a file path.
    :param color_dict: Dictionary. Represents colors to be used in overlays.
    :return: None.
    """
    # opening image
    open_img = imread(image_path)
    open_img = cvtColor(open_img, COLOR_BGR2RGB)

    # getting image data from df
    current_image_df = merged_df[merged_df['img_file_name'] == image_name]

    # filtering current image data based on detection threshold
    current_image_df = current_image_df[current_image_df['detection_threshold'] >= detection_threshold]

    # getting current image detection/annotation counts
    detections = current_image_df[current_image_df['evaluator'] == 'model']
    annotations = current_image_df[current_image_df['evaluator'] == 'fornma']
    detection_count = len(detections)
    annotation_count = len(annotations)

    # defining base texts
    model_text = f'model: {detection_count}'
    fornma_text = f'fornma: {annotation_count}'
    threshold_text = f'DT: {detection_threshold}'

    # adding base texts to image corner
    putText(open_img,
            model_text,
            (10, 30),
            FONT_HERSHEY_SIMPLEX,
            0.9,
            color_dict['model'],
            2)
    putText(open_img,
            fornma_text,
            (10, 60),
            FONT_HERSHEY_SIMPLEX,
            0.9,
            color_dict['fornma'],
            2)
    putText(open_img,
            threshold_text,
            (10, 90),
            FONT_HERSHEY_SIMPLEX,
            0.9,
            color_dict['DT'],
            2)

    # adding overlays to image
    add_multiple_overlays(open_img=open_img,
                          current_image_df=current_image_df,
                          color_dict=color_dict)

    # saving image in output path
    open_img = cvtColor(open_img, COLOR_RGB2BGR)
    imwrite(output_path, open_img)


def add_overlays_to_multiple_images(input_folder: str,
                                    images_extension: str,
                                    detection_file_path: str,
                                    ground_truth_file_path: str or None,
                                    output_folder: str,
                                    detection_threshold: float,
                                    color_dict: dict
                                    ) -> None:
    """
    Given a path to a folder containing images,
    a path to a file containing detection info from
    mentioned images, and a path to an output folder,
    generates images with detection outlines (if centroid
    flag is False, or centroids, if flag is True), saving
    new outlined images into output folder.
    :param input_folder: String. Represents a folder path.
    :param images_extension: String. Represents image extension.
    :param detection_file_path: String. Represents a file path.
    :param ground_truth_file_path: String. Represents a file path.
    :param output_folder: String. Represents a folder path.
    :param detection_threshold: Float. Represents detection threshold to be applied as filter.
    :param color_dict: Dictionary. Represents colors to be used in overlays.
    :return: None.
    """
    # getting merged detections df
    merged_df = get_merged_detection_annotation_df(detections_df_path=detection_file_path,
                                                   annotations_df_path=ground_truth_file_path)

    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=input_folder,
                                          extension=images_extension)
    images_num = len(images)
    images_names = [image_name.replace(images_extension, '')
                    for image_name
                    in images]

    # iterating over images_names
    for image_index, image_name in enumerate(images_names, 1):

        # printing execution message
        progress_ratio = image_index / images_num
        progress_percentage = progress_ratio * 100
        progress_percentage_round = round(progress_percentage)
        progress_string = f'adding overlays to image {image_index} of {images_num} ({progress_percentage_round}%)'
        flush_or_print(string=progress_string,
                       index=image_index,
                       total=images_num)

        # getting image path
        image_name_w_extension = f'{image_name}{images_extension}'
        image_path = join(input_folder, image_name_w_extension)

        # getting output path
        output_name = f'{image_name}_overlays.png'
        output_path = join(output_folder, output_name)

        # adding overlays to current image
        add_overlays_to_single_image(image_name=image_name,
                                     image_path=image_path,
                                     merged_df=merged_df,
                                     detection_threshold=detection_threshold,
                                     output_path=output_path,
                                     color_dict=color_dict)

    # printing execution message
    f_string = f'overlays added to all {images_num} images!'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting detection file path
    detection_file = args_dict['detection_file']

    # getting ground-truth file path
    ground_truth_file = args_dict['ground_truth_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting detection threshold
    detection_threshold = args_dict['detection_threshold']
    detection_threshold = float(detection_threshold)

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running add_overlays_to_multiple_images function
    add_overlays_to_multiple_images(input_folder=input_folder,
                                    images_extension=images_extension,
                                    detection_file_path=detection_file,
                                    ground_truth_file_path=ground_truth_file,
                                    output_folder=output_folder,
                                    detection_threshold=detection_threshold,
                                    color_dict=COLOR_DICT)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
