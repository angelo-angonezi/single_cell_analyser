# generate detection metrics df module

print('initializing...')  # noqa

# Code destined to generating data frame containing
# info on TP, FP, FN, precision, recall and F1-Score
# for each image in test set.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from numpy import arange
from pandas import concat
from pandas import Series
from numpy import ndarray
from pandas import DataFrame
from numpy import add as np_add
from numpy import count_nonzero
from argparse import ArgumentParser
from numpy import zeros as np_zeroes
from src.utils.aux_funcs import get_etc
from src.utils.aux_funcs import draw_circle
from src.utils.aux_funcs import get_time_str
from src.utils.aux_funcs import draw_ellipse
from src.utils.aux_funcs import draw_rectangle
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import get_current_time
from src.utils.aux_funcs import get_time_elapsed
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import simple_hungarian_algorithm
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

START = 0.0
STOP = 1.0
IOU_STEP = 0.05
DETECTION_STEP = 0.1
IOU_RANGE = arange(START,
                   STOP + IOU_STEP,
                   IOU_STEP)
DETECTION_RANGE = arange(START,
                         STOP + DETECTION_STEP,
                         DETECTION_STEP)
IOU_THRESHOLDS = [round(i, 2) for i in IOU_RANGE]
DETECTION_THRESHOLDS = [round(i, 2) for i in DETECTION_RANGE]

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate detection metrics df module'

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


def get_pixel_mask(row_data: Series,
                   style: str
                   ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with respective style overlay.
    :param row_data: Series. Represents OBB coords data.
    :param style: String. Represents an overlay style.
    :return: ndarray. Represents base image with mask overlay.
    """
    # extracting coords from row data
    cx = int(row_data['cx'])
    cy = int(row_data['cy'])
    width = float(row_data['width'])
    height = float(row_data['height'])
    angle = float(row_data['angle'])

    # defining color (same for all styles)
    color = (1,)

    # defining base image
    base_img = get_blank_image()

    # checking mask style
    if style == 'ellipse':

        # adding elliptical mask
        draw_ellipse(open_img=base_img,
                     cx=cx,
                     cy=cy,
                     width=width,
                     height=height,
                     angle=angle,
                     color=color,
                     thickness=-1)

    elif style == 'circle':

        # getting radius
        radius = (width + height) / 2
        radius = int(radius)

        # adding circular mask
        draw_circle(open_img=base_img,
                    cx=cx,
                    cy=cy,
                    radius=radius,
                    color=color,
                    thickness=-1)

    elif style == 'rectangle':

        # adding rectangular mask
        draw_rectangle(open_img=base_img,
                       cx=cx,
                       cy=cy,
                       width=width,
                       height=height,
                       angle=angle,
                       color=color,
                       thickness=-1)

    # returning modified image
    return base_img


def get_cost_matrix(detections_df: DataFrame,
                    annotations_df: DataFrame,
                    style: str
                    ) -> ndarray:
    """
    Given a detections df and an annotations df
    from a same image, returns cost matrix based
    on given style.
    """
    # creating a placeholder cost matrix
    cost_matrix = np_zeroes([len(detections_df),
                             len(annotations_df)])

    # getting detections/annotations rows
    detections_rows = detections_df.iterrows()
    annotations_rows = annotations_df.iterrows()

    # calculating IoU score for each detection-annotation pair

    # iterating over lines (detections)
    for line_index, detection_row_info in enumerate(detections_rows):

        # getting detection row data
        _, detection_row_data = detection_row_info

        # getting detection mask
        detection_mask = get_pixel_mask(row_data=detection_row_data,
                                        style=style)

        # iterating over columns (annotations)
        for column_index, annotation_row_info in enumerate(annotations_rows):

            # getting annotation row data
            _, annotation_row_data = annotation_row_info

            # getting detection mask
            annotation_mask = get_pixel_mask(row_data=annotation_row_data,
                                             style=style)

            # calculating IoU between two rows
            current_iou = get_iou(mask_a=detection_mask,
                                  mask_b=annotation_mask)

            # getting opposite iou score (needed to apply hungarian algorithm)
            current_cost = 1 - current_iou

            # updating cost matrix
            cost_matrix[line_index][column_index] = current_cost

    # returning cost matrix
    return cost_matrix


def get_image_metrics(df: DataFrame,
                      detection_threshold: float,
                      iou_threshold: float,
                      style: str
                      ) -> tuple:
    """
    Given a merged detections/annotations data frame,
    returns a tuple of metrics, determined by given
    style, of following structure:
    (true_positives, false_positives, false_negatives)
    """
    # defining placeholder values for tp, fp, fn
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # converting iou threshold to cost threshold (same is done for cost matrix)
    iou_threshold_cost = 1 - iou_threshold

    # getting current image detections/annotations dfs
    detections_df = df[df['evaluator'] == 'model']
    annotations_df = df[df['evaluator'] == 'fornma']

    # filtering detections_df by detection_threshold
    detections_df = detections_df[detections_df['detection_threshold'] > detection_threshold]

    # getting current image detections/annotations counts
    detections_num = len(detections_df)
    annotations_num = len(annotations_df)

    # checking detections/annotations counts
    if detections_num > annotations_num:

        # calculating difference
        diff = detections_num - annotations_num

        # updating false positive count
        false_positives += diff

    if annotations_num > detections_num:

        # calculating difference
        diff = annotations_num - detections_num

        # updating false negative count
        false_negatives += diff

    # getting current image cost matrix
    cost_matrix = get_cost_matrix(detections_df=detections_df,
                                  annotations_df=annotations_df,
                                  style=style)

    # establishing relations between detections/annotations using hungarian algorithm
    detections_indices, annotations_indices = simple_hungarian_algorithm(cost_matrix=cost_matrix)

    # getting indices zip
    indices_zip = zip(detections_indices, annotations_indices)

    # iterating over indices zip
    for index_zip in indices_zip:

        # getting detection/annotation index
        detection_index, annotation_index = index_zip

        # retrieving detection/annotation cost
        current_cost = cost_matrix[detection_index][annotation_index]

        # checking whether current cost is below threshold

        # if it is below cost threshold
        if current_cost < iou_threshold_cost:

            # updating true positives count
            true_positives += 1

        # if it exceeds cost threshold
        else:

            # updating false positives count
            false_positives += 1

            # updating false negatives count
            false_negatives += 1

    # TODO: remove these lines once test completed
    # print()
    # f_string = f'Detections count: {detections_num}\n'
    # f_string += f'Annotations count: {annotations_num}\n'
    # f_string += f'Established cells (detections): {len(detections_indices)}\n'
    # f_string += f'Established cells (annotations): {len(annotations_indices)}\n'
    # f_string += f'True Positives: {true_positives}\n'
    # f_string += f'False Negatives: {false_negatives}\n'
    # f_string += f'False Positives: {false_positives}'
    # print(f_string)
    # input()

    # assembling final tuple
    metrics_tuple = (true_positives, false_positives, false_negatives)

    # returning final tuple
    return metrics_tuple


def create_detection_metrics_df(df: DataFrame,
                                iou_thresholds: list,
                                detection_thresholds: list,
                                style: str
                                ) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a data frame containing true positive,
    false positive and false negative counts,
    based on given style IoU+Hungarian Algorithm
    matching of detections.
    """
    # getting start time
    start_time = get_current_time()

    # defining placeholder value for dfs_list
    dfs_list = []

    # grouping df by images
    image_groups = df.groupby('img_file_name')

    # getting totals
    iou_thresholds_num = len(iou_thresholds)
    detection_thresholds_num = len(detection_thresholds)
    images_num = len(image_groups)
    iterations_total = images_num * iou_thresholds_num * detection_thresholds_num

    # defining starter for current iteration
    current_iteration = 1

    # iterating over image groups
    for image_index, (image_name, image_group) in enumerate(image_groups, 1):

        # iterating over IoU thresholds
        for iou in iou_thresholds:

            # iterating over detection thresholds
            for dt in detection_thresholds:

                # getting current progress
                progress_ratio = current_iteration / iterations_total
                progress_percentage = progress_ratio * 100

                # getting current time
                current_time = get_current_time()

                # getting time elapsed
                time_elapsed = get_time_elapsed(start_time=start_time,
                                                current_time=current_time)

                # getting estimated time of completion
                etc = get_etc(time_elapsed=time_elapsed,
                              current_iteration=current_iteration,
                              iterations_total=iterations_total)

                # converting times to adequate format
                time_elapsed_str = get_time_str(time_in_seconds=time_elapsed)
                etc_str = get_time_str(time_in_seconds=etc)

                # defining progress string
                progress_string = f'analysing image {image_index}/{images_num} '
                progress_string += f'| IoU: {iou:02.1f} '
                progress_string += f'| DT: {dt:02.1f} '
                progress_string += f'| progress: {progress_percentage:02.2f}% '
                progress_string += f'| time elapsed: {time_elapsed_str} '
                progress_string += f'| ETC: {etc_str}'
                progress_string += '   '

                # printing execution message
                flush_or_print(string=progress_string,
                               index=current_iteration,
                               total=iterations_total)

                # getting current image metrics
                tp, fp, fn = get_image_metrics(df=image_group,
                                               detection_threshold=dt,
                                               iou_threshold=iou,
                                               style=style)

                # calculating precision
                try:
                    precision = tp / (tp + fp)
                except ZeroDivisionError:
                    precision = 0

                # calculating recall
                try:
                    recall = tp / (tp + fn)
                except ZeroDivisionError:
                    recall = 0

                # calculating f1_score
                try:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                except ZeroDivisionError:
                    f1_score = 0

                # getting current image dict
                current_dict = {'img_name': image_name,
                                'mask_style': style,
                                'iou_threshold': iou,
                                'detection_threshold': dt,
                                'true_positives': tp,
                                'false_positives': fp,
                                'false_negatives': fn,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f1_score}

                # getting current image df
                current_df = DataFrame(current_dict,
                                       index=[0])

                # appending current df to dfs_list
                dfs_list.append(current_df)

                # updating current iteration
                current_iteration += 1

    # concatenating dfs in dfs_list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def generate_detection_metrics_df(fornma_file: str,
                                  detections_file: str,
                                  output_path: str,
                                  iou_thresholds: list,
                                  detection_thresholds: list,
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
    :param output_path: String. Represents a path to a file.
    :param iou_thresholds: List. Represents a list of IoU thresholds.
    :param detection_thresholds: List. Represents a list of detection thresholds.
    :param style: String. Represents overlays style (rectangle/circle/ellipse).
    :return: None.
    """
    # getting merged detections/annotations df
    print('getting merged detections/annotations df...')
    merged_df = get_merged_detection_annotation_df(detections_df_path=detections_file,
                                                   annotations_df_path=fornma_file)

    # getting detection metrics df
    print('creating detection metrics df...')
    detection_metrics_df = create_detection_metrics_df(df=merged_df,
                                                       iou_thresholds=iou_thresholds,
                                                       detection_thresholds=detection_thresholds,
                                                       style=style)

    # saving detection metrics df
    print('saving detection metrics df...')
    detection_metrics_df.to_csv(output_path,
                                index=False)

    # printing execution message
    print(f'output saved to "{output_path}"')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """
    Gets execution parameters from
    command line and runs main function.
    """
    # getting args dict
    args_dict = get_args_dict()

    # getting fornma file
    fornma_file = args_dict['fornma_file']

    # getting detections file
    detections_file = args_dict['detections_file']

    # getting output path
    output_path = args_dict['output_path']

    # getting mask style
    mask_style = args_dict['mask_style']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_detection_metrics_df function
    generate_detection_metrics_df(fornma_file=fornma_file,
                                  detections_file=detections_file,
                                  output_path=output_path,
                                  iou_thresholds=IOU_THRESHOLDS,
                                  detection_thresholds=DETECTION_THRESHOLDS,
                                  style=mask_style)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
