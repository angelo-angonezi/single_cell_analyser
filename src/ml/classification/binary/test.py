# binary classification test module

print('initializing...')  # noqa

# Code destined to testing binary
# classification neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os import environ
from pandas import merge
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

# setting tensorflow warnings off
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'classification test module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # annotations file param
    parser.add_argument('-a', '--annotations-file',
                        dest='annotations_file',
                        required=True,
                        help='defines path to ANNOTATED crops info df (.csv) file')

    # predictions file param
    parser.add_argument('-p', '--predictions-file',
                        dest='predictions_file',
                        required=True,
                        help='defines path to prediction df (.csv) file')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def load_df(data_file: str,
            data_type: str
            ) -> DataFrame:
    """
    Given a path to a annotations file,
    returns filtered df containing
    required cols for analysis.
    """
    # reading df
    df = read_csv(data_file)

    # defining cols to keep
    cols_to_keep = ['crop_name', 'class']

    # dropping unrequired cols
    df = df[cols_to_keep]

    # checking data type
    if data_type == 'prediction':

        # renaming cols
        cols = ['crop_name', 'prediction']
        df.columns = cols

    # returning filtered df
    return df


def get_metrics(test_df: DataFrame,
                predictions_df: DataFrame
                ) -> tuple:
    """
    Given a test and predictions
    dfs, calculates metrics and
    returns metrics.
    """
    # merging dfs by crop_name
    merged_df = merge(left=test_df,
                      right=predictions_df,
                      on='crop_name')
    print(merged_df)

    # getting instances count
    df_len = len(merged_df)

    # getting possible classes
    classes_col = merged_df['class']
    classes_col_list = classes_col.to_list()
    classes_set = set(classes_col_list)
    classes_list = list(classes_set)
    possible_classes = sorted(classes_list)

    # getting classes df/num/proportions
    negative_class = possible_classes[1]
    positive_class = possible_classes[0]
    negative_class_df = merged_df[merged_df['class'] == negative_class]
    positive_class_df = merged_df[merged_df['class'] == positive_class]
    negative_class_num = len(negative_class_df)
    positive_class_num = len(positive_class_df)
    negative_class_ratio = negative_class_num / df_len
    positive_class_ratio = positive_class_num / df_len
    negative_class_percentage = negative_class_ratio * 100
    positive_class_percentage = positive_class_ratio * 100
    negative_class_percentage_round = round(negative_class_percentage)
    positive_class_percentage_round = round(positive_class_percentage)

    # printing classes legend
    f_string = '--Classes legend--\n'
    f_string += f'"Negative" class: {negative_class} ({negative_class_num} | {negative_class_percentage_round}%)\n'
    f_string += f'"Positive" class: {positive_class} ({positive_class_num} | {positive_class_percentage_round}%)\n'
    f_string += '------------------'
    print(f_string)

    # defining placeholder values for tp, tn, fp, fn
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # getting rows num
    rows_num = len(merged_df)

    # getting df rows
    df_rows = merged_df.iterrows()

    # defining starter for current_row_index
    current_row_index = 1

    # iterating over fornma df rows
    for row_index, row_data in df_rows:

        # printing progress message
        base_string = 'checking prediction #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row class/prediction
        current_class = row_data['class']
        current_prediction = row_data['prediction']

        # checking if class/prediction match

        # if both belong to positive class
        if current_class == positive_class and current_prediction == positive_class:

            # updating true positives
            true_positives += 1

        # if both belong to negative class
        if current_class == negative_class and current_prediction == negative_class:

            # updating true negatives
            true_negatives += 1

        # if class is positive, and prediction is negative
        if current_class == positive_class and current_prediction == negative_class:

            # updating false negatives
            false_negatives += 1

        # if class is negative, and prediction is positive
        if current_class == negative_class and current_prediction == positive_class:

            # updating false positives
            false_positives += 1

        # updating current row index
        current_row_index += 1

    # assembling final tuple
    metrics_tuple = (true_positives,
                     true_negatives,
                     false_positives,
                     false_negatives)

    # returning final tuple
    return metrics_tuple


def binary_classification_test(annotations_file: str,
                               predictions_file: str
                               ) -> None:
    """
    Given a path to an annotated data df,
    and a path to a file containing test
    data predictions, prints metrics
    on console.
    """
    # getting test df
    print('getting test df...')
    test_df = load_df(data_file=annotations_file,
                      data_type='test')

    # getting predictions df
    print('getting predictions df...')
    predictions_df = load_df(data_file=predictions_file,
                             data_type='prediction')

    # getting images num
    images_num = len(predictions_df)

    # getting metrics
    print('getting base metrics...')
    metrics = get_metrics(test_df=test_df,
                          predictions_df=predictions_df)
    tp, tn, fp, fn = metrics

    # calculating other metrics
    print('calculating other metrics...')
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (tpr + tnr) / 2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    precision_plus_recall = precision + recall
    precision_times_recall = precision * recall
    f1_score = 2 * (precision_times_recall / precision_plus_recall)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # rounding values
    print('rounding values...')
    tpr = round(tpr, 2)
    tnr = round(tnr, 2)
    fpr = round(fpr, 2)
    fnr = round(fnr, 2)
    accuracy = round(accuracy, 2)
    balanced_accuracy = round(balanced_accuracy, 2)
    precision = round(precision, 2)
    recall = round(recall, 2)
    f1_score = round(f1_score, 2)
    sensitivity = round(sensitivity, 2)
    specificity = round(specificity, 2)

    # printing metrics on console
    print('printing metrics...')
    f_string = f'---Metrics Results---\n'
    f_string += f'Test images num: {images_num}\n'
    f_string += f'TP: {tp}\n'
    f_string += f'TN: {tn}\n'
    f_string += f'FP: {fp}\n'
    f_string += f'FN: {fn}\n'
    f_string += f'TPR: {tpr}\n'
    f_string += f'TNR: {tnr}\n'
    f_string += f'FPR: {fpr}\n'
    f_string += f'FNR: {fnr}\n'
    f_string += f'Accuracy: {accuracy}\n'
    f_string += f'Balanced Accuracy: {balanced_accuracy}\n'
    f_string += f'Precision: {precision}\n'
    f_string += f'Recall: {recall}\n'
    f_string += f'F1-Score: {f1_score}\n'
    f_string += f'Sensitivity: {sensitivity}\n'
    f_string += f'Specificity: {specificity}\n'
    f_string += '---------end---------'
    print(f_string)

    # printing execution message
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting annotations file param
    annotations_file = args_dict['annotations_file']

    # predictions file param
    predictions_file = args_dict['predictions_file']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    using_gpu = is_using_gpu()
    using_gpu_str = f'Using GPU: {using_gpu}'
    print(using_gpu_str)

    # waiting for user input
    enter_to_continue()

    # running binary_classification_test function
    binary_classification_test(annotations_file=annotations_file,
                               predictions_file=predictions_file)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
