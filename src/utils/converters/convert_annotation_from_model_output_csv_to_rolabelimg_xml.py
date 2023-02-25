# annotation format converter (model output csv to rolabelimg xml)

# annotation format conversion module (from model output to rolabelimg format)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
from cv2 import imread
from math import radians
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import flush_or_print

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert annotations from alpr to xml format"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    input_help = 'defines path to folder containing images (required to get image dimensions)'
    parser.add_argument('-i', '--input-images',
                        dest='input_images',
                        type=str,
                        required=True,
                        help=input_help)

    detections_help = 'defines path to csv file containing model output detections (normal and mitoses txts joined)'
    parser.add_argument('-d', '--detection-file',
                        dest='detections_file',
                        type=str,
                        required=True,
                        help=detections_help)

    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        type=str,
                        required=True,
                        help='defines output folder (folder that will contain rolabelimg [.xml] annotations)')

    # detection threshold param
    parser.add_argument('-t', '--detection-threshold',
                        dest='detection_threshold',
                        type=float,
                        required=True,
                        help='defines detection threshold')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def convert_single_file(img_name: str,
                        img_dataframe: DataFrame,
                        output_xml_file_path: str,
                        img_width: int,
                        img_height: int,
                        ) -> None:
    """
    Given a path to a csv file containing annotations in alpr format,
    converts annotations to rolabelimg annotation format, saving a xml file in
    given output path.
    :param img_name: String. Represents image name.
    :param img_dataframe: DataFrame. Represents detections for a single image.
    :param output_xml_file_path: String. Represents a path to a xml file.
    :param img_width: Integer. Represents image width in pixels.
    :param img_height: Integer. Represents image height in pixels.
    :return: None.
    """
    # creating placeholder value for xml output string
    output_lines = ''

    # adding starting line for xml output string
    output_lines += '<annotation verified="no">'
    output_lines += '<folder>FolderName</folder>'

    # adding file name line to xml output string
    output_lines += f'<filename>{img_name}</filename>'

    # adding file path line to xml output string
    output_lines += f'<path>{img_name}.png</path>'

    # adding source database to xml output string
    output_lines += '<source>'
    output_lines += '<database>Unknown</database>'
    output_lines += '</source>'

    # adding size to xml output string
    output_lines += f'<size>'
    output_lines += f'<width>{img_width}</width>'
    output_lines += f'<height>{img_height}</height>'
    output_lines += f'<depth>1</depth>'
    output_lines += f'</size>'

    # adding segmentation line to xml output string
    output_lines += '<segmented>0</segmented>'

    # iterating over current image bounding boxes
    for bbox in img_dataframe.iterrows():

        # getting index and bbox data
        bbox_index, bbox_data = bbox

        # getting bbox info
        bbox_cx = bbox_data['cx']
        bbox_cy = bbox_data['cy']
        bbox_width = bbox_data['width']
        bbox_height = bbox_data['height']
        bbox_angle = bbox_data['angle']
        bbox_angle = radians(bbox_angle)
        bbox_class = bbox_data['class']

        # adding object lines to xml output string
        output_lines += '<object>'
        output_lines += '<type>robndbox</type>'
        output_lines += f'<name>{bbox_class}</name>'
        output_lines += '<pose>Unspecified</pose>'
        output_lines += '<truncated>0</truncated>'
        output_lines += '<difficult>0</difficult>'
        output_lines += '<robndbox>'
        output_lines += f'<cx>{bbox_cx}</cx>'
        output_lines += f'<cy>{bbox_cy}</cy>'
        output_lines += f'<w>{bbox_width}</w>'
        output_lines += f'<h>{bbox_height}</h>'
        output_lines += f'<angle>{bbox_angle}</angle>'
        output_lines += '</robndbox>'
        output_lines += '</object>'

    # adding final line to xml output string
    output_lines += '</annotation>'

    # converting string format
    converted_lines = output_lines.encode(encoding='UTF-8', errors='strict')

    # writing output file
    with open(output_xml_file_path, 'wb') as open_output_file:

        # adding lines to file
        open_output_file.write(converted_lines)


def convert_multiple_files(images_folder_path: str,
                           input_alpr_file_path: str,
                           output_xml_folder: str,
                           detection_threshold: float
                           ) -> None:
    """
    Given a path to a folder containing multiple xmls containing annotations
    in rolabelimg format, converts annotations to alpr format, saving outputs
    as csv files in given output folder.
    :param images_folder_path: String. Represents a path to an image.
    :param input_alpr_file_path: String. Represents a path to a folder containing csv files.
    :param output_xml_folder: String. Represents a path to save output files.
    :param detection_threshold: Float. Represents a detection threshold.
    :return: None.
    """
    # opening input csv
    input_data = read_csv(input_alpr_file_path)

    # filtering detections based on detection threshold filter
    filtered_data = input_data.loc[input_data['detection_threshold'] >= detection_threshold]

    # grouping input dataframe by image name
    img_groups = filtered_data.groupby('img_file_name')
    imgs_num = len(img_groups)

    # iterating over image groups
    for img_index, (img_name, img_group) in enumerate(img_groups, 1):

        # getting image path
        image_name_w_extension = f'{img_name}.tif'
        image_path = join(images_folder_path,
                          image_name_w_extension)

        # opening image
        current_img = imread(image_path)

        # getting current image dimensions
        try:
            img_size = current_img.shape
        except AttributeError:
            f_string = f'Unable to get image "{image_path}"'
            f_string += '.Using 1280x720 as default img dimensions.'
            print(f_string)
            img_size = (720, 1280, 3)

        img_height, img_width, _ = img_size

        # getting output path
        output_name = f'{img_name}.xml'
        output_path = join(output_xml_folder, output_name)

        # printing execution message
        progress_ratio = img_index / imgs_num
        progress_percentage = progress_ratio * 100
        progress_percentage_round = round(progress_percentage)
        f_string = f'converting annotations for file {img_index} of {imgs_num} ({progress_percentage_round}%)'
        flush_or_print(string=f_string,
                       index=img_index,
                       total=imgs_num)

        # running single converter function
        convert_single_file(img_name=img_name,
                            img_dataframe=img_group,
                            output_xml_file_path=output_path,
                            img_width=img_width,
                            img_height=img_height)

    # printing execution message
    f_string = f'all {imgs_num} annotation files generated!'
    print('')
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input images folder
    input_file = args_dict['input_images']

    # getting detections file
    detections_file = args_dict['detections_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting detection threshold
    detection_threshold = args_dict['detection_threshold']

    # running multiple converter function
    convert_multiple_files(images_folder_path=input_file,
                           input_alpr_file_path=detections_file,
                           output_xml_folder=output_folder,
                           detection_threshold=detection_threshold)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
