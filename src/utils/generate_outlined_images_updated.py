#!/usr/bin/python3

# given a path to a folder containing images
# and a path to a detection output results file
# and a path to an output folder,
# generates images with OBBs overlay,
# displaying detection thresholds ranges as different colors.

######################################################################
# imports

# importing required libraries

import cv2
import numpy as np
from os import listdir
from sys import stdout
from os.path import join
from pandas import read_csv
from argparse import ArgumentParser

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "generate outlined images\n"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines input folder (folder containing images)')

    # image extension param
    parser.add_argument('-x', '--image-extension',
                        dest='image_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folder')

    # detection file param
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help='defines path to file containing detection info')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines output folder (folder that will contain outlined images)')

    # output folder param
    parser.add_argument('-t', '--detection-threshold',
                        dest='detection_threshold',
                        required=False,
                        default=0.6,
                        help='defines threshold to be applied (filters detections OBBs based on detection threshold)')

    # centroid option param
    parser.add_argument('-C', '--centroid',
                        dest='centroid_flag',
                        action='store_true',
                        required=False,
                        default=False,
                        help='defines whether to draw outline (default) or centroid (if current flag is passed)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


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


def generate_outlined_images(input_folder: str,
                             image_extension: str,
                             detection_file_path: str,
                             output_folder: str,
                             detection_threshold: float,
                             centroid_flag: bool
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
    :param output_folder: String. Represents a folder path.
    :param detection_threshold: Float. Represents detection threshold to be applied as filter.
    :param centroid_flag: Boolean. Represents a True/False flag.
    :return: None.
    """
    # printing execution message
    print('reading detection file...')

    # reading detections file
    detections_df = read_csv(detection_file_path)

    # grouping detections by image name
    df_grouped_by_img_name = detections_df.groupby(['img_file_name'])

    # printing execution message
    groups_num = len(df_grouped_by_img_name)
    f_string = f'detection info found for {groups_num} images'
    print(f_string)

    # iterating over grouped detections
    for img_index, (name, img_name_group) in enumerate(df_grouped_by_img_name, 1):

        # getting image path
        image_name = f'{name}{image_extension}'
        image_path = join(input_folder, image_name)

        # printing execution message
        percentage_progress = img_index * 100 / groups_num
        round_percentage_progress = round(percentage_progress)
        f_string = f'adding overlays to image {img_index} of {groups_num}'
        f_string += f' ({round_percentage_progress}%)'

        if img_index == groups_num:
            print(f_string)
        else:
            flush_string(f_string)

        # opening image
        open_img = cv2.imread(image_path)
        open_img = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)

        # iterating over detections
        for detection_index, detection in img_name_group.iterrows():

            # getting current detection bounding box info
            current_detection_threshold = detection['detection_threshold']
            cx = detection['cx']
            cy = detection['cy']
            width = detection['width']
            height = detection['height']
            angle = detection['angle']
            det_class = detection['class']

            # adding threshold limit
            if current_detection_threshold < detection_threshold:
                continue

            # defining color for outline
            color = (0, 0, 0)
            if det_class == 'normal':
                if detection_threshold < 0.3:
                    color = (20, 30, 250)  # dark blue
                elif 0.3 <= detection_threshold < 0.6:
                    color = (40, 230, 230)  # cyan
                elif detection_threshold >= 0.6:
                    color = (40, 230, 20)  # green
            elif det_class == 'round':
                if detection_threshold < 0.3:
                    color = (250, 5, 5)  # red
                elif 0.3 <= detection_threshold < 0.6:
                    color = (250, 140, 20)  # orange
                elif detection_threshold >= 0.6:
                    color = (240, 230, 20)  # yellow

            # checking centroid flag
            if centroid_flag:

                # adding centroid overlay
                draw_circle(img=open_img,
                            cx=cx,
                            cy=cy,
                            color=color)

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

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting detection threshold
    detection_threshold = args_dict['detection_threshold']
    detection_threshold = float(detection_threshold)

    # getting centroid flag
    centroid_flag = args_dict['centroid_flag']

    # printing execution parameters
    spacer = '_' * 50
    print(spacer)
    f_string = f'--Execution parameters--\n'
    f_string += f'input folder: {input_folder}\n'
    f_string += f'image extension: {image_extension}\n'
    f_string += f'detection file: {detection_file}\n'
    f_string += f'output folder: {output_folder}\n'
    f_string += f'detection threshold: {detection_threshold}\n'
    f_string += f'overlay: {"centroid" if centroid_flag else "rectangle"}'
    print(f_string)
    print(spacer)
    input('press "enter" to continue')

    # running generate_outlined_images function
    generate_outlined_images(input_folder=input_folder,
                             image_extension=image_extension,
                             detection_file_path=detection_file,
                             output_folder=output_folder,
                             detection_threshold=detection_threshold,
                             centroid_flag=centroid_flag)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
