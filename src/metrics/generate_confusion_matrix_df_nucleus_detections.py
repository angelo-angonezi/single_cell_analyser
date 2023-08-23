# generate confusion matrix df for nucleus detection module

print('initializing...')  # noqa

# Code destined to generating data frame containing
# info on TP, FP, and FN for each image in test set.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from numpy import ndarray
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from numpy import zeros as np_zeroes
from src.utils.aux_funcs import draw_circle
from src.utils.aux_funcs import draw_ellipse
from src.utils.aux_funcs import draw_rectangle
from src.utils.aux_funcs import get_crop_pixels
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate confusion matrix df for nucleus detection'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # fornma file param
    parser.add_argument('-f', '--fornma-file',
                        dest='fornma_file',
                        required=True,
                        help='defines path to fornma (.csv) file')

    # detections file param
    parser.add_argument('-d', '--detections-file',
                        dest='detections_file',
                        required=True,
                        help='defines path to detections (.csv) file')

    # detection threshold param
    parser.add_argument('-dt', '--detection-threshold',
                        dest='detection_threshold',
                        required=False,
                        default=0.5,
                        help='defines detection threshold to be applied as filter in model detections')

    # iou threshold param
    parser.add_argument('-it', '--iou-threshold',
                        dest='iou_threshold',
                        required=False,
                        default=0.5,
                        help='defines IoU threshold to be applied as filter in model detections')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder (which will contain output .csvs)')

    # style param
    parser.add_argument('-s', '--mask-style',
                        dest='mask_style',
                        required=False,
                        default='ellipse',
                        help='defines overlay style (rectangle/circle/ellipse)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_blank_image(width: int = 1408,
                    height: int = 1040
                    ) -> ndarray:
    """
    Given an image width/height, returns
    numpy array of given dimension
    filled with zeroes.
    :param width: Integer. Represents an image width.
    :param height: Integer. Represents an image height.
    :return: ndarray. Represents an image.
    """
    # defining matrix shape
    shape = (height, width)

    # creating blank matrix
    blank_matrix = np_zeroes(shape=shape)

    # returning blank matrix
    return blank_matrix


def get_pixel_mask(base_img: ndarray,
                   cx: int,
                   cy: int,
                   width: float,
                   height: float,
                   angle: float,
                   style: str
                   ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with respective style overlay.
    :param base_img: ndarray. Represents a base image.
    :param cx: Integer. Represents a coordinate in the cartesian plane.
    :param cy: Integer. Represents a coordinate in the cartesian plane.
    :param width: Float. Represents an OBB width.
    :param height: Float. Represents an OBB width.
    :param angle: Float. Represents an OBB angle.
    :param style: String. Represents an overlay style.
    :return: ndarray. Represents base image with mask overlay.
    """
    # checking mask style
    if style == 'ellipse':

        # adding elliptical mask
        draw_ellipse(open_img=base_img,
                     cx=cx,
                     cy=cy,
                     width=width,
                     height=height,
                     angle=angle,
                     color=(1,))

    # returning modified image
    return base_img


def add_pixel_mask_col(df: DataFrame,
                       style: str
                       ) -> None:
    """
    Given a merged detections/annotations
    data frame, adds "pixel_mask" column,
    based on OBBs coordinates.
    :param df: DataFrame. Represents a merged detections/annotations df.
    :param style: String. Represents overlays style (rectangle/circle/ellipse).
    :return: None.
    """
    # defining column name
    pixel_mask_col = 'pixel_mask'

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current_row_index
    current_row_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        base_string = 'adding pixel mask to nucleus #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # updating current_row_index
        current_row_index += 1

        # getting current row info
        cx = int(row_data['cx'])
        cy = int(row_data['cy'])
        width = float(row_data['width'])
        height = float(row_data['height'])
        angle = float(row_data['angle'])

        # creating blank array
        base_img = get_blank_image()

        from numpy import unique

        print(base_img)
        print(unique(base_img))

        # adding current detection/annotation to array
        pixel_mask = get_pixel_mask(base_img=base_img,
                                    cx=cx,
                                    cy=cy,
                                    width=width,
                                    height=height,
                                    angle=angle,
                                    style=style)

        print(pixel_mask)
        print(cx, cy)
        print(unique(pixel_mask))
        a = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\ml\\nucleus_detection\\test_results\\confusion_matrix\\a.png'
        from cv2 import imwrite
        imwrite(filename=a,
                img=pixel_mask)
        exit()
        # updating current row pixel mask column
        df.at[row_index, pixel_mask_col] = pixel_mask
        exit()


def generate_confusion_matrix_df(fornma_file: str,
                                 detections_file: str,
                                 detection_threshold: float,
                                 iou_threshold: float,
                                 output_folder: str,
                                 style: str
                                 ) -> None:
    """
    Given a path to detections and annotations files,
    generates a data frame containing info on TP, FP
    and FN counts based on eIoU+Hungarian Algorithm,
    in order to enable further metrics calculations,
    saving it to given output path.
    :param fornma_file: String. Represents a path to a file.
    :param detections_file: String. Represents a path to a file.
    :param detection_threshold: Float. Represents a detection threshold.
    :param iou_threshold: Float. Represents an IoU threshold.
    :param output_folder: String. Represents a path to a folder.
    :param style: String. Represents overlays style (rectangle/circle/ellipse).
    :return: None.
    """
    # getting merged detections/annotations df
    merged_df = get_merged_detection_annotation_df(detections_df_path=detections_file,
                                                   annotations_df_path=fornma_file)
    
    # adding pixels mask column to df
    add_pixel_mask_col(df=merged_df,
                       style='ellipse')

    print(merged_df)
    print(merged_df.columns)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting fornma file
    fornma_file = args_dict['fornma_file']

    # getting detections file
    detections_file = args_dict['detections_file']

    # getting detection threshold
    detection_threshold = args_dict['detection_threshold']

    # getting iou threshold
    iou_threshold = args_dict['iou_threshold']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting mask style
    mask_style = args_dict['mask_style']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running generate_confusion_matrix_df function
    generate_confusion_matrix_df(fornma_file=fornma_file,
                                 detections_file=detections_file,
                                 detection_threshold=detection_threshold,
                                 iou_threshold=iou_threshold,
                                 output_folder=output_folder,
                                 style=mask_style)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
