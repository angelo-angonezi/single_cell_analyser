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
                        required=True,
                        help='defines detection threshold to be applied as filter in model detections')

    # iou threshold param
    parser.add_argument('-it', '--iou-threshold',
                        dest='iou_threshold',
                        required=True,
                        help='defines IoU threshold to be applied as filter in model detections')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder (which will contain output .csvs)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_blank_image(width: float,
                    height: float
                    ) -> ndarray:
    pass


def generate_confusion_matrix_df(fornma_file: str,
                                 detections_file: str,
                                 detection_threshold: float,
                                 iou_threshold: float,
                                 output_folder: str,
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
    :return: None.
    """
    # getting merged detections/annotations df
    merged_df = get_merged_detection_annotation_df(detections_df_path=detections_file,
                                                   annotations_df_path=fornma_file)

    print(merged_df)

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

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_confusion_matrix_df function
    generate_confusion_matrix_df(fornma_file=fornma_file,
                                 detections_file=detections_file,
                                 detection_threshold=detection_threshold,
                                 iou_threshold=iou_threshold,
                                 output_path=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
