# classification test module

print('initializing...')  # noqa

# Code destined to testing
# classification neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from math import sqrt
from pandas import merge
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

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

    # dataset file param
    parser.add_argument('-d', '--dataset-file',
                        dest='dataset_file',
                        required=True,
                        help='defines path to dataset df (.csv) file')

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


def get_test_df(dataset_file: str) -> DataFrame:
    """
    Given a path to a dataset file,
    returns filtered df containing
    only test data and required cols
    for analysis.
    """
    # reading dataset df
    dataset_df = read_csv(dataset_file)

    # filtering df by test data
    filtered_df = dataset_df[dataset_df['split'] == 'test']

    # defining cols to keep
    cols_to_keep = ['crop_name',
                    'class']

    # dropping unrequired cols
    filtered_df = filtered_df[cols_to_keep]

    # returning filtered df
    return filtered_df


def get_predictions_df(predictions_file: str) -> DataFrame:
    """
    Given a path to a predictions file,
    returns loaded df filtered by cols
    related to test analysis.
    """
    # reading predictions df
    predictions_df = read_csv(predictions_file)

    # returning predictions df
    return predictions_df


def get_metrics(test_df: DataFrame,
                predictions_df: DataFrame
                ) -> tuple:
    """
    Given a test and predictions
    dfs, calculates metrics and
    returns metrics.
    """
    # joining dfs by crop_name
    joined_df = merge(left=test_df,
                      right=predictions_df,
                      on='crop_name')

    # adding placeholder value for prediction type col (TP, TN, FP, FN)
    joined_df['prediction_type'] = None

    # getting possible classes
    classes_col = joined_df['class']
    classes_col_list = classes_col.to_list()
    classes_set = set(classes_col_list)
    classes_list = list(classes_set)
    possible_classes = sorted(classes_list)

    # printing classes legend
    f_string = '--Classes legend--\n'
    f_string += f'"Positive" class: {possible_classes[0]}\n'
    f_string += f'"Negative" class: {possible_classes[1]}'
    print(f_string)

    # defining placeholder values for tp, tn, fp, fn
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # getting rows num
    rows_num = len(joined_df)

    # getting df rows
    df_rows = joined_df.iterrows()

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

        # if both belong to first class
        if current_class == possible_classes[0] and current_prediction == possible_classes[0]:

            # updating true positives
            true_positives += 1

        # if both belong to second class
        if current_class == possible_classes[1] and current_prediction == possible_classes[1]:

            # updating true negatives
            true_negatives += 1

        # if class is first, and prediction is second
        if current_class == possible_classes[0] and current_prediction == possible_classes[1]:

            # updating false negatives
            false_negatives += 1

        # if class is second, and prediction is first
        if current_class == possible_classes[1] and current_prediction == possible_classes[0]:

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


def classification_test(dataset_file: str,
                        predictions_file: str
                        ) -> None:
    """
    Given a path to dataset df, and
    a path to a file containing test
    data predictions, prints metrics
    on console.
    """
    # getting test df
    print('getting test df...')
    test_df = get_test_df(dataset_file=dataset_file)

    # getting images num
    images_num = len(test_df)

    # getting predictions df
    print('getting predictions df...')
    predictions_df = get_predictions_df(predictions_file=predictions_file)

    # getting metrics
    print('getting metrics...')
    metrics = get_metrics(test_df=test_df,
                          predictions_df=predictions_df)
    tp, tn, fp, fn = metrics

    # calculating other metrics
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (tpr + tnr) / 2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    precision_plus_recall = precision + recall
    precision_times_recall = precision * recall
    f1_score = 2 * (precision_times_recall / precision_plus_recall)

    # rounding values
    tpr = round(tpr, 2)
    tnr = round(tnr, 2)
    accuracy = round(accuracy, 2)
    balanced_accuracy = round(balanced_accuracy, 2)
    precision = round(precision, 2)
    recall = round(recall, 2)
    f1_score = round(f1_score, 2)

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
    f_string += f'Accuracy: {accuracy}\n'
    f_string += f'Precision: {precision}\n'
    f_string += f'Recall: {recall}\n'
    f_string += f'F1-Score: {f1_score}\n'
    f_string += f'Balanced Accuracy: {balanced_accuracy}'
    print(f_string)

    # printing execution message
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting dataset file param
    dataset_file = args_dict['dataset_file']

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

    # running classification_test function
    classification_test(dataset_file=dataset_file,
                        predictions_file=predictions_file)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
