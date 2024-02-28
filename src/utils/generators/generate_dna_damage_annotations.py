# generate dna damage annotations module

print('initializing...')  # noqa

# Code destined to generating
# dna damage annotations file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_crops_df
from src.utils.aux_funcs import get_contours_df
from src.utils.aux_funcs import get_crop_pixels
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_dna_damage_level
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

FOCI_COUNT_THRESHOLD = 5
FOCI_AREA_MEAN_THRESHOLD = 5.5

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate dna damage annotations module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # crops file param
    parser.add_argument('-c', '--crops-file',
                        dest='crops_file',
                        required=True,
                        help='defines path to crops file (crops_info.csv file)')

    # foci masks folder param
    parser.add_argument('-f', '--foci-masks-folder',
                        dest='foci_masks_folder',
                        required=True,
                        help='defines path to folder containing foci segmentation masks')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output file (.csv)')

    # foci min area param
    parser.add_argument('-fm', '--foci-min-area',
                        dest='foci_min_area',
                        required=False,
                        default=2,
                        help='defines minimum area for a foci (in pixels)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def add_dna_damage_col(df: DataFrame,
                       foci_masks_folder: str,
                       images_extension: str,
                       foci_min_area: str
                       ) -> None:
    """
    Given a crops info df, and paths to
    foci masks, adds dna damage col, based
    on foci count/area.
    """
    # defining col name
    col_name = 'class'

    # emptying class col
    df[col_name] = None

    # getting rows num
    rows_num = len(df)

    # getting df rows
    df_rows = df.iterrows()

    # defining starter for current row index
    current_row_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing progress message
        base_string = 'adding dna damage col to row #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row crop name
        crop_name = row_data['crop_name']

        # getting current row crop name with extension
        crop_name_w_extension = f'{crop_name}{images_extension}'

        # getting current row crop foci mask path
        foci_mask_path = join(foci_masks_folder,
                              crop_name_w_extension)

        # getting current crop contours df
        contours_df = get_contours_df(image_name=crop_name,
                                      image_path=foci_mask_path,
                                      contour_type='53bp1_foci')

        # filtering df by min foci area
        contours_df = contours_df[contours_df['area'] >= foci_min_area]

        # getting current crop foci count/area
        foci_count = len(contours_df)
        foci_area_col = contours_df['area']
        foci_area_mean = foci_area_col.mean()

        # getting current crop autophagy level
        current_class = get_dna_damage_level(foci_count=foci_count,
                                             foci_area_mean=foci_area_mean,
                                             foci_count_threshold=FOCI_COUNT_THRESHOLD,
                                             foci_area_mean_threshold=FOCI_AREA_MEAN_THRESHOLD)

        # updating current row data
        df.at[row_index, col_name] = current_class

        # updating current row index
        current_row_index += 1

def generate_dna_damage_annotations(crops_file: str,
                                    foci_masks_folder: str,
                                    images_extension: str,
                                    output_path: str,
                                    foci_min_area: int
                                    ) -> None:
    """
    Given a path to a folder containing foci
    masks, and a path to a file containing crops
    info, generates dna damage annotations, and
    saves it to given output path.
    """
    # reading crops info df
    crops_info_df = read_csv(crops_file)

    # updating class col
    add_dna_damage_col(df=crops_info_df,
                       foci_masks_folder=foci_masks_folder,
                       images_extension=images_extension,
                       foci_min_area=foci_min_area)

    # saving updated crops info df
    crops_info_df.to_csv(output_path,
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

    # getting crops file
    crops_file = args_dict['crops_file']

    # getting foci folder path
    foci_masks_folder = args_dict['foci_masks_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output path
    output_path = args_dict['output_path']

    # getting foci min area
    foci_min_area = args_dict['foci_min_area']
    foci_min_area = int(foci_min_area)

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_dna_damage_annotations function
    generate_dna_damage_annotations(crops_file=crops_file,
                                    foci_masks_folder=foci_masks_folder,
                                    images_extension=images_extension,
                                    output_path=output_path,
                                    foci_min_area=foci_min_area)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
