# annotation format converter (crops info csv to fornma csv)

# annotation format conversion module (from crops info to fornma)
# Code destined to converting annotation formats for ML applications.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_mask_area
from src.utils.aux_funcs import get_axis_ratio
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_pixel_intensity
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_nucleus_pixel_intensity
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

IMAGE_NAME_COL = 'Image_name_merge'
RED_MEAN_COL = 'Mean_red'
GREEN_MEAN_COL = 'Mean_green'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert annotations from crops info to fornma"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input path param
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help='defines path to crops info df (.csv) file')

    # red folder param
    parser.add_argument('-r', '--red-folder',
                        dest='red_folder',
                        required=False,
                        help='defines red input folder (folder containing crops in fluorescence channel)')

    # green folder param
    parser.add_argument('-g', '--green-folder',
                        dest='green_folder',
                        required=False,
                        help='defines green input folder (folder containing crops in fluorescence channel)')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=False,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output fornma format (.csv) file')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def convert_single_file(input_csv_file_path: str,
                        red_folder: str,
                        green_folder: str,
                        images_extension: str,
                        output_path: str
                        ) -> None:
    """
    Given a path to a crops info file,
    converts annotations to fornma format,
    saving results in given output path.
    """
    # opening csv file
    print('reading input file...')
    crops_df = read_csv(input_csv_file_path)

    # defining placeholder value for dfs list
    dfs_list = []

    # getting rows num
    rows_num = len(crops_df)

    # getting fornma df rows
    rows = crops_df.iterrows()

    # defining starter for current_row_index
    current_row_index = 1

    # iterating over fornma df rows
    for row_index, row_data in rows:

        # flushing/printing execution message
        f_string = f'getting info on nucleus #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row cols
        current_img_name = row_data['img_name']
        current_img_min = row_data['img_min']
        current_img_max = row_data['img_max']
        current_crop_index = row_data['crop_index']
        current_crop_name = row_data['crop_name']
        current_cx = row_data['cx']
        current_cy = row_data['cy']
        current_width = row_data['width']
        current_height = row_data['height']

        # adding extension to crop name
        current_crop_name_w_extension = f'{current_crop_name}{images_extension}'

        # checking whether user passed images to get min/max
        if red_folder is not None:

            # getting current crop channel paths
            current_red_path = join(red_folder,
                                    current_crop_name_w_extension)
            current_green_path = join(green_folder,
                                      current_crop_name_w_extension)

            # getting current crop mean pixel intensities
            # TODO: check whether to use mean or median here
            red_mean = get_nucleus_pixel_intensity(crop_path=current_red_path,
                                                   nucleus_width=current_width,
                                                   nucleus_height=current_height,
                                                   calc='mean')
            green_mean = get_nucleus_pixel_intensity(crop_path=current_green_path,
                                                     nucleus_width=current_width,
                                                     nucleus_height=current_height,
                                                     calc='mean')

            # normalizing intensities
            red_mean_normalized = (red_mean - current_img_min) / current_img_max
            green_mean_normalized = (green_mean - current_img_min) / current_img_max

        # getting current crop area
        current_area = get_mask_area(row_data=row_data,
                                     style='ellipse')

        # getting current crop NII
        current_nii = get_axis_ratio(width=current_width,
                                     height=current_height)

        # checking whether user passed images to get min/max
        if red_folder is not None:

            # creating current obb dict
            current_obb_dict = {'Cell': current_crop_index,
                                'X': current_cx,
                                'Y': current_cy,
                                'Area': current_area,
                                'NII': current_nii,
                                'Image_name_merge': current_img_name,
                                'Mean_red': red_mean_normalized,
                                'Mean_green': green_mean_normalized}

        else:

            # creating current obb dict
            current_obb_dict = {'Cell': current_crop_index,
                                'X': current_cx,
                                'Y': current_cy,
                                'Area': current_area,
                                'NII': current_nii,
                                'Image_name_merge': current_img_name}

        # creating current obb df
        current_obb_df = DataFrame(current_obb_dict,
                                   index=[0])

        # appending current obb df to dfs list
        dfs_list.append(current_obb_df)

        # updating current_row_index
        current_row_index += 1

    # concatenating dfs in dfs list
    print('assembling final df...')
    final_df = concat(dfs_list)

    # saving final df in output path
    print('saving output file...')
    final_df.to_csv(output_path,
                    index=False)

    # printing execution message
    f_string = f'all fornma results successfully converted!\n'
    f_string += f'results saved at "{output_path}"'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # getting red folder
    red_folder = args_dict['red_folder']

    # getting green folder
    green_folder = args_dict['green_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running converter function
    convert_single_file(input_csv_file_path=input_file,
                        red_folder=red_folder,
                        green_folder=green_folder,
                        images_extension=images_extension,
                        output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
