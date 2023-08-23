# generate confusion matrix df for nucleus detection module

print('initializing...')  # noqa

# Code destined to generating data frame containing
# info on TP, FP, and FN for each image in test set.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imwrite
from os.path import join
from pandas import concat
from numpy import ndarray
from pandas import read_csv
from pandas import DataFrame
from numpy import add as np_add
from numpy import count_nonzero
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

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output (.csv) file')

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


def get_iou(mask_a: ndarray,
            mask_b: ndarray
            ) -> float:
    """
    Given two pixel masks, representing
    detected/annotated OBBs, returns IoU.
    :param mask_a: ndarray. Represents a pixel mask.
    :param mask_b: ndarray. Represents a pixel mask.
    :return: Float. Represents an IoU value.
    """
    # adding arrays
    final_array = np_add(mask_a, mask_b)

    # counting "1" pixels (just one of the masks cover)
    one_count = count_nonzero(final_array == 1)

    # counting "2" pixels (= intersection -> both masks cover)
    two_count = count_nonzero(final_array == 2)

    # getting intersection
    intersection = two_count

    # getting union
    union = one_count + two_count

    # calculating IoU (Intersection over Union)
    iou_value = intersection / union

    # returning IoU
    return iou_value


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
                     color=(1,),
                     thickness=-1)

    # returning modified image
    return base_img


def create_confusion_matrix_df(df: DataFrame,
                               detection_threshold: float,
                               iou_threshold: float,
                               style: str
                               ) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a data frame containing true positive,
    false positive and false negative counts,
    based on given style IoU+Hungarian Algorithm
    matching of detections.
    """
    pass


def generate_confusion_matrix_df(fornma_file: str,
                                 detections_file: str,
                                 detection_threshold: float,
                                 iou_threshold: float,
                                 output_path: str,
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
    :param output_path: String. Represents a path to a file.
    :param style: String. Represents overlays style (rectangle/circle/ellipse).
    :return: None.
    """
    # getting merged detections/annotations df
    merged_df = get_merged_detection_annotation_df(detections_df_path=detections_file,
                                                   annotations_df_path=fornma_file)
    
    # getting confusion matrix df
    confusion_matrix_df = create_confusion_matrix_df(df=merged_df,
                                                     detection_threshold=detection_threshold,
                                                     iou_threshold=iou_threshold,
                                                     style=style)
    
    # saving confusion matrix df
    confusion_matrix_df.to_csv(output_path,
                               index=False)

    # printing execution message
    print(f'output saved to {output_path}')
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

    # getting output path
    output_path = args_dict['output_path']

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
                                 output_path=output_path,
                                 style=mask_style)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
