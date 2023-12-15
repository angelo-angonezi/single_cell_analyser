# generate detection metrics df module

print('initializing...')  # noqa

# Code destined to generating data frame containing
# info on TP, FP, FN, precision, recall and F1-Score
# for each image in test set.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import line
from cv2 import imread
from cv2 import imwrite
from os.path import join
from pandas import concat
from numpy import ndarray
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from numpy import zeros as np_zeroes
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import get_iou
from src.utils.aux_funcs import get_etc
from src.utils.aux_funcs import get_time_str
from src.utils.aux_funcs import get_mask_area
from src.utils.aux_funcs import get_pixel_mask
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import get_current_time
from src.utils.aux_funcs import get_time_elapsed
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_test_images_df
from src.utils.aux_funcs import get_image_confluence
from src.utils.aux_funcs import get_experiment_well_df
from src.utils.aux_funcs import simple_hungarian_algorithm
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

# thresholds lists
IOU_THRESHOLDS = [0.5]
DETECTION_THRESHOLDS = [0.5]

# progress bar related
CURRENT_IOU = 0
IOUS_TOTAL = len(IOU_THRESHOLDS)
CURRENT_DT = 0
DTS_TOTAL = len(DETECTION_THRESHOLDS)
CURRENT_ITERATION = 1
ITERATIONS_TOTAL = 0
CURRENT_IMAGE = 1
IMAGES_TOTAL = 0
START_TIME = 0

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

    # input folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        type=str,
                        help='defines path to folder containing images',
                        required=True)

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        type=str,
                        help='defines extension (.tif, .png, .jpg) of images in input folder',
                        required=True)

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

    # lines and treatment file param
    parser.add_argument('-lt', '--lines-treatment-file',
                        dest='lines_treatment_file',
                        required=True,
                        help='defines path to csv file containing info on cell lines and treatments.')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder')

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


def get_iterations_total(df: DataFrame,
                         iou_thresholds: list,
                         detection_thresholds: list,
                         ) -> int:
    """
    Returns total iterations number,
    based on given parameters.
    """
    # defining placeholder value for iterations total
    iterations_total = 0

    # grouping df by images
    image_groups = df.groupby('img_file_name')

    # iterating over image groups
    for image_index, (image_name, image_group) in enumerate(image_groups, 1):

        # iterating over IoU thresholds
        for iou in iou_thresholds:

            # iterating over detection thresholds
            for dt in detection_thresholds:

                # getting current image detections/annotations dfs
                detections_df = image_group[image_group['evaluator'] == 'model']
                annotations_df = image_group[image_group['evaluator'] == 'fornma']

                # filtering detections_df by detection_threshold
                detections_df = detections_df[detections_df['detection_threshold'] > dt]

                # getting current image detections/annotations nums
                detections_num = len(detections_df)
                annotations_num = len(annotations_df)

                # getting current combinations number
                combinations_num = detections_num * annotations_num

                # updating iterations total
                iterations_total += combinations_num

    # returning iterations total
    return iterations_total


def print_global_progress():
    """
    Takes global parameters and prints
    progress message on console.
    """
    # getting global variables
    global CURRENT_IOU
    global IOUS_TOTAL
    global CURRENT_DT
    global DTS_TOTAL
    global CURRENT_ITERATION
    global ITERATIONS_TOTAL
    global CURRENT_IMAGE
    global IMAGES_TOTAL
    global START_TIME

    # getting current progress
    progress_ratio = CURRENT_ITERATION / ITERATIONS_TOTAL
    progress_percentage = progress_ratio * 100

    # getting current time
    current_time = get_current_time()

    # getting time elapsed
    time_elapsed = get_time_elapsed(start_time=START_TIME,
                                    current_time=current_time)

    # getting estimated time of completion
    etc = get_etc(time_elapsed=time_elapsed,
                  current_iteration=CURRENT_ITERATION,
                  iterations_total=ITERATIONS_TOTAL)

    # converting times to adequate format
    time_elapsed_str = get_time_str(time_in_seconds=time_elapsed)
    etc_str = get_time_str(time_in_seconds=etc)

    # defining progress string
    progress_string = f'analysing image {CURRENT_IMAGE}/{IMAGES_TOTAL}... '
    progress_string += f'| IoU: {CURRENT_IOU:02.1f} '
    progress_string += f'| DT: {CURRENT_DT:02.1f} '
    progress_string += f'| progress: {progress_percentage:02.2f}% '
    progress_string += f'| time elapsed: {time_elapsed_str} '
    progress_string += f'| ETC: {etc_str}'
    progress_string += '   '

    # printing execution message
    flush_or_print(string=progress_string,
                   index=CURRENT_ITERATION,
                   total=ITERATIONS_TOTAL)


def get_cost_matrix(detections_df: DataFrame,
                    annotations_df: DataFrame,
                    style: str
                    ) -> ndarray:
    """
    Given a detections df and an annotations df
    from a same image, returns cost matrix based
    on given style.
    """
    # getting annotations/detection num
    detections_num = len(detections_df)
    annotations_num = len(annotations_df)

    # creating a placeholder cost matrix
    cost_matrix = np_zeroes([detections_num,
                             annotations_num])

    # getting detections/annotations rows
    detections_df = detections_df.reset_index()
    annotations_df = annotations_df.reset_index()

    # calculating IoU score for each detection-annotation pair

    # iterating over lines (detections)
    for line_index in range(detections_num):

        # getting detection row data
        detection_row_data = detections_df.iloc[line_index]

        # getting detection mask
        detection_mask = get_pixel_mask(row_data=detection_row_data,
                                        style=style)

        # iterating over columns (annotations)
        for column_index in range(annotations_num):

            # printing execution message
            print_global_progress()

            # getting annotation row data
            annotation_row_data = annotations_df.iloc[column_index]

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

            # updating global variables
            global CURRENT_ITERATION
            CURRENT_ITERATION += 1

    # returning cost matrix
    return cost_matrix


def get_image_metrics(images_folder: str,
                      images_extension: str,
                      image_name: str,
                      output_folder: str,
                      detections_df: DataFrame,
                      annotations_df: DataFrame,
                      iou_threshold: float,
                      style: str
                      ) -> tuple:
    """
    Given a merged detections/annotations data frame,
    returns a tuple of metrics, determined by given
    style, of following structure:
    (true_positives, false_positives, false_negatives)
    """
    # defining placeholder values for tp, fp, fn, area_errors
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    area_errors = []

    # converting iou threshold to cost threshold (same is done for cost matrix)
    iou_threshold_cost = 1 - iou_threshold

    # getting current image path
    image_name_w_extension = f'{image_name}{images_extension}'
    image_path = join(images_folder,
                      image_name_w_extension)

    # opening current image
    open_img = imread(image_path)

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

            # retrieving current detection coords/area
            detection_row = detections_df.iloc[detection_index]
            detection_cx = int(detection_row['cx'])
            detection_cy = int(detection_row['cy'])
            detection_coords = (detection_cx, detection_cy)
            detection_area = get_mask_area(row_data=detection_row,
                                           style=style)

            # retrieving current annotation coords/area
            annotation_row = annotations_df.iloc[annotation_index]
            annotation_cx = int(annotation_row['cx'])
            annotation_cy = int(annotation_row['cy'])
            annotation_coords = (annotation_cx, annotation_cy)
            annotation_area = get_mask_area(row_data=annotation_row,
                                            style=style)

            # getting current area error
            current_area_error = detection_area - annotation_area

            # appending current area error to area error list
            area_errors.append(current_area_error)

            # drawing line between detection/annotation coords
            line(open_img,
                 annotation_coords,
                 detection_coords,
                 (0, 0, 255),
                 1)

        # if it exceeds cost threshold
        else:

            # updating false positives count
            false_positives += 1

            # updating false negatives count
            false_negatives += 1

    # Lines to check TP, FP, FN obtaining
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

    # getting global variables
    global CURRENT_IOU
    global CURRENT_DT

    # saving associations overlays image
    save_name = image_name_w_extension.replace('.png', f'_iou{CURRENT_IOU}_dt{CURRENT_DT}.png')
    save_path = join(output_folder,
                     save_name)
    imwrite(save_path,
            open_img)

    # assembling final tuple
    metrics_tuple = (true_positives,
                     false_positives,
                     false_negatives,
                     area_errors)

    # returning final tuple
    return metrics_tuple


def create_detection_metrics_df(images_folder: str,
                                images_extension: str,df: DataFrame,
                                lines_treatment_df: DataFrame,
                                iou_thresholds: list,
                                detection_thresholds: list,
                                style: str,
                                output_folder: str
                                ) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a data frame containing true positive,
    false positive and false negative counts,
    based on given style IoU+Hungarian Algorithm
    matching of detections.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # grouping df by images
    image_groups = df.groupby('img_file_name')

    # getting images total
    images_total = len(image_groups)

    # updating global variables
    global IMAGES_TOTAL
    global START_TIME
    IMAGES_TOTAL = images_total
    START_TIME = get_current_time()

    # iterating over image groups
    for image_index, (image_name, image_group) in enumerate(image_groups, 1):

        if image_index == 3:
            break

        # iterating over IoU thresholds
        for iou in iou_thresholds:

            # updating global variables
            global CURRENT_IOU
            CURRENT_IOU = iou

            # iterating over detection thresholds
            for dt in detection_thresholds:

                # updating global variables
                global CURRENT_DT
                CURRENT_DT = dt

                # getting current image name split
                image_name_split = image_name.split('_')

                # getting image experiment string list
                experiment_split = image_name_split[:-4]

                # getting image experiment
                current_experiment = '_'.join(experiment_split)

                # getting current image well
                current_well = image_name_split[-4]

                # getting current image field
                current_field = image_name_split[-3]

                # getting current lines and treatments df row
                current_lines_treatments_df_row = get_experiment_well_df(df=lines_treatment_df,
                                                                         experiment=current_experiment,
                                                                         well=current_well)

                # getting current author
                current_author = current_lines_treatments_df_row['author']

                # getting current image cell line
                current_cell_line = current_lines_treatments_df_row['cell_line']

                # getting current image treatment
                current_treatment = current_lines_treatments_df_row['treatment']

                # getting current image detections/annotations dfs
                detections_df = image_group[image_group['evaluator'] == 'model']
                annotations_df = image_group[image_group['evaluator'] == 'fornma']

                # filtering detections_df by detection_threshold
                detections_df = detections_df[detections_df['detection_threshold'] > dt]

                # getting current image detections/annotations nums
                detections_num = len(detections_df)
                annotations_num = len(annotations_df)

                # getting current image metrics
                tp, fp, fn, area_errors = get_image_metrics(images_folder=images_folder,
                                                            images_extension=images_extension,
                                                            image_name=image_name,
                                                            output_folder=output_folder,
                                                            detections_df=detections_df,
                                                            annotations_df=annotations_df,
                                                            iou_threshold=iou,
                                                            style=style)

                # getting current image confluences
                model_confluence = get_image_confluence(df=detections_df,
                                                        style=style)
                fornma_confluence = get_image_confluence(df=annotations_df,
                                                         style=style)

                # calculating precision
                # TODO: Check whether setting metrics to zero when
                #  this error occurs makes sense!
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
                                'author': current_author,
                                'cell_line': current_cell_line,
                                'treatment': current_treatment,
                                'well': current_well,
                                'field': current_field,
                                'mask_style': style,
                                'model_count': detections_num,
                                'fornma_count': annotations_num,
                                'model_confluence': model_confluence,
                                'fornma_confluence': fornma_confluence,
                                'iou_threshold': iou,
                                'detection_threshold': dt,
                                'area_errors': area_errors,
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

    # concatenating dfs in dfs_list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def generate_detection_metrics_df(images_folder: str,
                                  images_extension: str,
                                  fornma_file: str,
                                  detections_file: str,
                                  lines_treatment_file: str,
                                  output_folder: str,
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
    """
    # getting merged detections/annotations df
    print('getting merged detections/annotations df...')
    merged_df = get_merged_detection_annotation_df(detections_df_path=detections_file,
                                                   annotations_df_path=fornma_file)

    # updating global variables
    global ITERATIONS_TOTAL
    print('getting iterations total...')
    ITERATIONS_TOTAL = get_iterations_total(df=merged_df,
                                            iou_thresholds=iou_thresholds,
                                            detection_thresholds=detection_thresholds)

    # reading lines treatment file
    print('getting lines/treatments df...')
    lines_treatment_df = read_csv(lines_treatment_file)

    # dropping train images (contain only fornma as evaluator)
    print('getting test images only...')
    filtered_df = get_test_images_df(df=merged_df)

    # getting detection metrics df
    print('creating detection metrics df...')
    detection_metrics_df = create_detection_metrics_df(images_folder=images_folder,
                                                       images_extension=images_extension,
                                                       df=filtered_df,
                                                       lines_treatment_df=lines_treatment_df,
                                                       iou_thresholds=iou_thresholds,
                                                       detection_thresholds=detection_thresholds,
                                                       style=style,
                                                       output_folder=output_folder)

    # saving detection metrics df
    print('saving detection metrics df...')
    save_path = join(output_folder,
                     'metrics_df.pickle')
    detection_metrics_df.to_pickle(save_path,
                                   index=False)

    # printing execution message
    print(f'output saved to "{output_folder}".')
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

    # getting images folder
    images_folder = args_dict['images_folder']

    # getting images extension
    images_extension = args_dict['images_extension']

    # getting fornma file
    fornma_file = args_dict['fornma_file']

    # getting detections file
    detections_file = args_dict['detections_file']

    # getting lines and treatment file param
    lines_treatment_file = args_dict['lines_treatment_file']

    # getting output path
    output_folder = args_dict['output_folder']

    # getting mask style
    mask_style = args_dict['mask_style']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)
    print('IoUs:', IOU_THRESHOLDS)
    print('DTs:', DETECTION_THRESHOLDS)
    spacer()

    # waiting for user input
    enter_to_continue()

    # running generate_detection_metrics_df function
    generate_detection_metrics_df(images_folder=images_folder,
                                  images_extension=images_extension,
                                  fornma_file=fornma_file,
                                  detections_file=detections_file,
                                  lines_treatment_file=lines_treatment_file,
                                  output_folder=output_folder,
                                  iou_thresholds=IOU_THRESHOLDS,
                                  detection_thresholds=DETECTION_THRESHOLDS,
                                  style=mask_style)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
