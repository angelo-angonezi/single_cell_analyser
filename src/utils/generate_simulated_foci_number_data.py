# generate simulated data module

print('initializing...')  # noqa

# code destined to generating random
# simulated data for ml codes testing only.

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
from time import sleep
from os import listdir
from random import randint
from pandas import DataFrame
from argparse import ArgumentParser
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate simulated data for single-cell crops ML'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser
    parser.add_argument('-i', '--images-input-folder',
                        dest='images_input_folder',
                        type=str,
                        help='defines path to folder containing images',
                        required=True)

    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        type=str,
                        help='defines path to output csv file containing simulated data',
                        required=True)

    parser.add_argument('-n', '--foci-min',
                        dest='foci_num_min',
                        type=int,
                        help='defines minimum number of foci in simulated data',
                        required=True)

    parser.add_argument('-m', '--foci-max',
                        dest='foci_num_max',
                        type=int,
                        help='defines maximum number of foci in simulated data',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def simulate_foci_data(images_folder_path: str,
                       output_path: str,
                       foci_num_min: int,
                       foci_num_max: int
                       ) -> None:
    """
    Given a path to a folder containing images,
    a path to an output file which will contain
    simulated data (csv format), and foci minimum
    and maximum numbers, generates simulated
    data for ml purposes.
    :param images_folder_path: String. Represents a path to a folder.
    :param output_path: String. Represents a path to a file.
    :param foci_num_min: Integer. Represents number of foci in a given image.
    :param foci_num_max: Integer. Represents number of foci in a given image.
    :return: None.
    """
    # printing execution message
    f_string = f'getting images in input folder...'
    print(f_string)

    # getting images in input folder
    images_in_folder = listdir(images_folder_path)

    # getting images names
    images_names = [img_name.replace('.tif', '')
                    for img_name
                    in images_in_folder
                    if img_name.endswith('.tif')]

    # getting number of images
    images_num = len(images_names)

    # printing execution message
    f_string = f'{images_num} images found.'
    print(f_string)

    # printing execution message
    f_string = f'creating random foci counts for images...'
    print(f_string)

    # creating random numbers list (min/max value determined by parameters)
    foci_nums = [randint(foci_num_min,
                         foci_num_max)
                 for _
                 in images_names]

    # creating output dictionary
    output_dict = {'image_name': images_names,
                   'foci_count': foci_nums}

    # creating output dataframe from dictionary
    output_df = DataFrame(output_dict)

    # saving final df in output path
    output_df.to_csv(output_path,
                     index=False)

    # printing execution message
    f_string = f'simulated foci counts data saved at "{output_path}"'
    print(f_string)

######################################################################
# defining main function


def main():
    """
    Runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting images input folder path
    images_folder_path = args_dict['images_input_folder']

    # getting output path
    output_path = args_dict['output_path']

    # getting foci number min
    foci_num_min = args_dict['foci_num_min']

    # getting foci number max
    foci_num_max = args_dict['foci_num_max']

    # running data simulation function
    simulate_foci_data(images_folder_path=images_folder_path,
                       output_path=output_path,
                       foci_num_min=foci_num_min,
                       foci_num_max=foci_num_max)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
