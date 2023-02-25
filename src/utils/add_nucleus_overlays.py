# add nucleus overlay module

print('initializing...')  # noqa

# Code destined to generating images with nucleus
# overlays based on model detections (and GT annotations).

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
import cv2
import numpy as np
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

COLOR_DICT = {'model': (),
              'fornma': ()}


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
    parser.add_argument('-x', '--image-extension',
                        dest='image_extension',
                        required=True,
                        help=extension_help)

    # detection file param
    detection_help = 'defines path to file containing model detections'
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help=detection_help)

    # gt file param
    gt_help = 'defines path to file containing ground-truth annotations '
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


def draw_rectangle(img,
                   cx: float,
                   cy: float,
                   width: float,
                   height: float,
                   angle: float,
                   color: tuple
                   ) -> None:
    """
    Given an open image, and corners for a rectangle with color,
    draws rectangle on base image.
    """
    # get the corner points
    box = cv2.boxPoints(((cx, cy),
                         (width, height),
                         angle))

    # converting corners format
    box = np.int0(box)

    # drawing lines
    cv2.drawContours(img, [box], -1, color, 2)


def draw_circle(img,
                cx: float,
                cy: float,
                color: tuple
                ) -> None:
    """
    Given an open image, and a set of cartesian
    coordinates, draws circle on base image.
    """
    # converting centroid coords to integers
    cx = int(cx)
    cy = int(cy)

    # defining radius and thickness
    radius = 3
    thickness = -1  # -1 fills circle

    # drawing circle
    cv2.circle(img, (cx, cy), radius, color, thickness)


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
    # getting image data from df
    current_image_df = merged_df[merged_df['img_file_name'] == image_name]

    # opening image
    open_img = cv2.imread(image_path)
    open_img = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)

    # getting df rows
    df_rows = current_image_df.iterrows()

    # iterating over df_rows
    for row in df_rows:

        # getting current row bounding box info
        current_detection_threshold = row['detection_threshold']
        cx = row['cx']
        cy = row['cy']
        width = row['width']
        height = row['height']
        angle = row['angle']
        det_class = row['class']
        evaluator = row['evaluator']

        # adding threshold limit
        if current_detection_threshold < detection_threshold:

            # skipping if current threshold is filtered
            continue

        # defining color for overlay
        overlay_color = color_dict[evaluator]


        else:

            # adding rectangle overlay
            draw_rectangle(img=open_img,
                           cx=cx,
                           cy=cy,
                           width=width,
                           height=height,
                           angle=angle,
                           color=color)

        # saving image in output folder
        output_name = f'{name}_outlined.png'
        output_path = join(output_folder, output_name)
        open_img = cv2.cvtColor(open_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, open_img)




def add_overlays_to_multiple_images(input_folder: str,
                                    image_extension: str,
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
    :param image_extension: String. Represents image extension.
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
                                          extension=image_extension)
    images_num = len(images)
    images_names = [image_name.replace(image_extension, '')
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
        image_name = f'{image_name}{image_extension}'
        image_path = join(input_folder, image_name)

        # getting current image detections/annotations
        current_image_detections =





    # printing execution message
    f_string = f'overlays added to all {groups_num} images!'
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
    image_extension = args_dict['image_extension']

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
    f_string = f'--Execution parameters--\n'
    f_string += f'input folder: {input_folder}\n'
    f_string += f'image extension: {image_extension}\n'
    f_string += f'detection file: {detection_file}\n'
    f_string += f'ground-truth file: {ground_truth_file}\n'
    f_string += f'output folder: {output_folder}\n'
    f_string += f'detection threshold: {detection_threshold}\n'
    f_string += f'overlay: {"centroid" if centroid_flag else "rectangle"}'
    spacer()
    print(f_string)
    spacer()
    input('press "Enter" to continue')

    # running generate_outlined_images function
    add_overlays_to_multiple_images(input_folder=input_folder,
                                    image_extension=image_extension,
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
