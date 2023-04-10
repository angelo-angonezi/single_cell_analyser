# annotation format converter (rolabelimg xml to alpr txt)

# annotation format conversion module (from rolabelimg to alpr-unconstrained format)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
import math
from os.path import join
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
    description = "convert annotations from xml to alpr format"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines input folder (folder containing rolabelimg[.xml] annotations)')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines output folder (folder that will contain alpr[.txt] annotations)')

    # class param
    class_help = 'if this flag is passed, detection class will be disregarded '
    class_help += '(all detection will be labeled as "Cell" only).'
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


def rotate(origin, point, angle):  # copied from stackoverflow
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def convert_single_file(input_xml_file_path: str,
                        output_txt_file_path: str,
                        disregard_class: bool
                        ) -> None:
    """
    Given a path to a xml file containing annotations in rolabelimg format,
    converts annotations to alpr annotation format, saving a txt file in
    given output path.
    :param input_xml_file_path: String. Represents a path to a xml file.
    :param output_txt_file_path: String. Represents a path to a txt file.
    :param disregard_class: Boolean. Represents a disregard class toggle.
    :return: None.
    """
    # opening xml file
    tree = ElementTree.parse(input_xml_file_path)
    root = tree.getroot()

    # defining placeholder values for width and height (updated below)
    images_width = 0.0
    images_height = 0.0

    # getting images size
    images_size = root.iter('size')

    for size in images_size:
        width = size.find('width').text
        height = size.find('height').text
        images_width = float(width)
        images_height = float(height)

    # getting bounding boxes and objects from xml file
    all_boxes = root.iter('robndbox')
    all_objs = root.iter('object')

    # creating zip for iteration
    box_obj_zip = zip(all_boxes, all_objs)

    # opening output file
    with open(output_txt_file_path, 'w') as open_output_file:

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
            angle_in_rads_float = float(angle_in_rads_text)

            # getting class value
            cell_class = obj.find('name')
            cell_class_text = cell_class.text
            cell_class_text = 'RoundCell' if cell_class_text == 'RoundCell' else 'NormalCell'

            # overwriting cell class value if disregard_class toggle is on
            if disregard_class:
                cell_class_text = 'Cell'

            # getting original coordinates
            origin = (cx_float, cy_float)

            # x1, y1
            untranslated_x1 = cx_float - (width_float / 2)
            untranslated_y1 = cy_float - (height_float / 2)
            untranslated_p1 = (untranslated_x1, untranslated_y1)
            translated_p1 = rotate(point=untranslated_p1,
                                   origin=origin,
                                   angle=angle_in_rads_float)
            translated_x1, translated_y1 = translated_p1

            # x2, y2
            untranslated_x2 = cx_float + (width_float / 2)
            untranslated_y2 = cy_float - (height_float / 2)
            untranslated_p2 = (untranslated_x2, untranslated_y2)
            translated_p2 = rotate(point=untranslated_p2,
                                   origin=origin,
                                   angle=angle_in_rads_float)
            translated_x2, translated_y2 = translated_p2

            # x3, y3
            untranslated_x3 = cx_float + (width_float / 2)
            untranslated_y3 = cy_float + (height_float / 2)
            untranslated_p3 = (untranslated_x3, untranslated_y3)
            translated_p3 = rotate(point=untranslated_p3,
                                   origin=origin,
                                   angle=angle_in_rads_float)
            translated_x3, translated_y3 = translated_p3

            # x4, y4
            untranslated_x4 = cx_float - (width_float / 2)
            untranslated_y4 = cy_float + (height_float / 2)
            untranslated_p4 = (untranslated_x4, untranslated_y4)
            translated_p4 = rotate(point=untranslated_p4,
                                   origin=origin,
                                   angle=angle_in_rads_float)
            translated_x4, translated_y4 = translated_p4

            # writing new line in output file

            new_line = f'4,'  # constant number of vertices
            new_line += f'{translated_x1 / images_width},'
            new_line += f'{translated_x2 / images_width},'
            new_line += f'{translated_x3 / images_width},'
            new_line += f'{translated_x4 / images_width},'
            new_line += f'{translated_y1 / images_height},'
            new_line += f'{translated_y2 / images_height},'
            new_line += f'{translated_y3 / images_height},'
            new_line += f'{translated_y4 / images_height},'
            new_line += f'{cell_class_text}'
            new_line += f'\n'
            open_output_file.write(new_line)


def convert_multiple_files(input_xml_folder: str,
                           output_txt_folder: str,
                           disregard_class: bool
                           ) -> None:
    """
    Given a path to a folder containing multiple xmls containing annotations
    in rolabelimg format, converts annotations to alpr format, saving outputs
    as txt files in given output folder.
    :param input_xml_folder: String. Represents a path to a folder containing xml files.
    :param output_txt_folder: String. Represents a path to save output files.
    :param disregard_class: Boolean. Represents a disregard class toggle.
    :return: None.
    """
    # getting xml files in input folder
    xml_files = get_specific_files_in_folder(path_to_folder=input_xml_folder,
                                             extension='.xml')
    num_of_files = len(xml_files)

    # iterating over files
    for index, file in enumerate(xml_files, 1):

        # getting input path
        input_path = join(input_xml_folder, file)

        # getting output path
        output_name = file.replace('.xml', '.txt')
        output_path = join(output_txt_folder, output_name)

        # running single converter function
        f_string = f'converting annotations for file #INDEX of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=index,
                               total=num_of_files)

        convert_single_file(input_xml_file_path=input_path,
                            output_txt_file_path=output_path,
                            disregard_class=disregard_class)

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

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting disregard class flag
    disregard_class = args_dict['disregard_class']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running multiple converter function
    convert_multiple_files(input_xml_folder=input_folder,
                           output_txt_folder=output_folder,
                           disregard_class=disregard_class)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
