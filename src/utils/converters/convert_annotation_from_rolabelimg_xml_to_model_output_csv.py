# annotation format converter (rolabelimg xml to model output csv)

# annotation format conversion module (from rolabelimg to model output format)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
import math
from os.path import join
from math import degrees
from pandas import concat
from pandas import DataFrame
from xml.etree import ElementTree
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert annotations from xml to model output format"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines input folder (folder containing rolabelimg[.xml] annotations)')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output file (.csv)')

    # class param
    class_help = 'if this flag is passed, detection class will be disregarded '
    class_help += '(all detection will be labeled as "normal" only).'
    parser.add_argument('-d', '--disregard-class',
                        dest='disregard_class',
                        action='store_true',
                        help=class_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def convert_single_file(file_name: str,
                        input_xml_file_path: str,
                        disregard_class: bool
                        ) -> DataFrame:
    """
    Given a path to a xml file containing annotations in rolabelimg format,
    converts annotations to model output format, returning respective
    data frame.
    :param file_name: String. Represents input file name.
    :param input_xml_file_path: String. Represents a path to a xml file.
    :param disregard_class: Boolean. Represents a disregard class toggle.
    :return: DataFrame. Represents converted annotations.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting image name
    image_name = file_name.replace('.xml', '')

    # opening xml file
    tree = ElementTree.parse(input_xml_file_path)
    root = tree.getroot()

    # getting bounding boxes and objects from xml file
    all_boxes = root.iter('robndbox')
    all_objs = root.iter('object')

    # creating zip for iteration
    box_obj_zip = zip(all_boxes, all_objs)

    # iterating over bounding boxes (to get coords) and objects (to get classes)
    for box_index, (box, obj) in enumerate(box_obj_zip, 1):

        # getting center x value
        cx = box.find('cx')
        cx_text = cx.text
        cx_float = float(cx_text)

        # getting center y value
        cy = box.find('cy')
        cy_text = cy.text
        cy_float = float(cy_text)

        # getting width value
        width = box.find('w')
        width_text = width.text
        width_float = float(width_text)

        # getting height value
        height = box.find('h')
        height_text = height.text
        height_float = float(height_text)

        # getting angle value
        angle = box.find('angle')
        angle_in_rads_text = angle.text
        # TODO: check if final angle has to be in rads or degrees!
        angle_in_rads_float = float(angle_in_rads_text)
        angle_in_degs_float = degrees(angle_in_rads_float)

        # getting class value
        cell_class = obj.find('name')
        cell_class_text = cell_class.text
        cell_class_text = 'RoundCell' if cell_class_text == 'RoundCell' else 'NormalCell'

        # overwriting cell class value if disregard_class toggle is on
        if disregard_class:
            cell_class_text = 'normal'

        # defining detection threshold
        detection_threshold = 1.0  # always one since coming from manual annotations

        # assembling current object dict
        current_dict = {'img_file_name': image_name,
                        'detection_threshold': detection_threshold,
                        'cx': cx_float,
                        'cy': cy_float,
                        'width': width_float,
                        'height': height_float,
                        'angle': angle_in_degs_float,
                        'class': cell_class_text}

        # assembling current object df
        current_df = DataFrame(current_dict,
                               index=[0])

        # appending current object df to dfs list
        dfs_list.append(current_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning df
    return final_df


def convert_multiple_files(input_xml_folder: str,
                           output_csv_path: str,
                           disregard_class: bool
                           ) -> None:
    """
    Given a path to a folder containing multiple xmls containing annotations
    in rolabelimg format, converts annotations to alpr format, saving outputs
    as txt files in given output folder.
    :param input_xml_folder: String. Represents a path to a folder containing xml files.
    :param output_csv_path: String. Represents a path to save output file.
    :param disregard_class: Boolean. Represents a disregard class toggle.
    :return: None.
    """
    # getting xml files in input folder
    xml_files = get_specific_files_in_folder(path_to_folder=input_xml_folder,
                                             extension='.xml')
    num_of_files = len(xml_files)

    # defining placeholder value for dfs list
    dfs_list = []

    # iterating over files
    for index, file in enumerate(xml_files, 1):

        # getting input path
        input_path = join(input_xml_folder, file)

        # running single converter function
        f_string = f'converting annotations for file #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=index,
                               total=num_of_files)

        # getting current image converted df
        current_df = convert_single_file(file_name=file,
                                         input_xml_file_path=input_path,
                                         disregard_class=disregard_class)

        # appending current image df to dfs list
        dfs_list.append(current_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # saving final df
    final_df.to_csv(output_csv_path,
                    index=False)

    # printing execution message
    f_string = f'all {num_of_files} annotation files converted!'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting output path
    output_path = args_dict['output_path']

    # getting disregard class flag
    disregard_class = args_dict['disregard_class']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running multiple converter function
    convert_multiple_files(input_xml_folder=input_folder,
                           output_csv_path=output_path,
                           disregard_class=disregard_class)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
