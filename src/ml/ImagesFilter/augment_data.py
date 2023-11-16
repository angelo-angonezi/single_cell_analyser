# ImagesFilter augment data module

print('initializing...')  # noqa

# Code destined to augmenting data for
# ImagesFilter classification network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import flip
from cv2 import imread
from cv2 import rotate
from cv2 import imwrite
from os.path import join
from cv2 import ROTATE_180
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'ImagesFilter data augmentation module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing images to be augmented.')

    # images extension param
    parser.add_argument('-e', '--extension',
                        dest='extension',
                        required=True,
                        help='defines images extension (.png, .jpg, .tif).')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder which will contain augmented images.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def augment_image(image_name: str,
                  images_folder: str,
                  output_folder: str
                  ) -> None:
    # getting current image path
    image_path = join(images_folder,
                      image_name)

    # opening current image
    open_image = imread(image_path)

    # resizing image
    from cv2 import resize
    from cv2 import INTER_AREA
    open_image = resize(open_image,
                        (512, 512),
                        interpolation=INTER_AREA)

    # rotating image
    rotated_image = rotate(open_image,
                           ROTATE_180)

    # flipping image vertically
    v_flipped_image = flip(open_image,
                           0)

    # flipping image horizontally
    h_flipped_image = flip(open_image,
                           1)

    # defining save path
    save_path = join(output_folder,
                     image_name)

    # saving images
    imwrite(save_path.replace('.jpg', '_o.jpg'), open_image)
    imwrite(save_path.replace('.jpg', '_r.jpg'), rotated_image)
    imwrite(save_path.replace('.jpg', '_vf.jpg'), v_flipped_image)
    imwrite(save_path.replace('.jpg', '_hf.jpg'), h_flipped_image)


def augment_data(images_folder: str,
                 extension: str,
                 output_folder: str
                 ) -> None:
    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=images_folder,
                                          extension=extension)

    # getting images num
    images_num = len(images)

    # setting number of modifications
    mods_num = 4
    final_imgs_num = images_num * mods_num

    # printing execution message
    f_string = f'found {images_num} images in input folder.'
    print(f_string)

    # defining placeholder value for current_image_index
    current_image_index = 1

    # iterating over images
    for image in images:

        # printing execution message
        base_string = 'augmenting image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_image_index,
                               total=images_num)

        # augmenting current image
        augment_image(image_name=image,
                      images_folder=images_folder,
                      output_folder=output_folder)

        # updating current_image_index
        current_image_index += 1

    # printing execution message
    print('augmentation complete!')
    print(f'final data set now contains {final_imgs_num} images.')
    print(f'results saved to "{output_folder}".')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder param
    images_folder = str(args_dict['images_folder'])

    # getting images extension param
    extension = str(args_dict['extension'])

    # getting output folder param
    output_folder = str(args_dict['output_folder'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running image_filter_test function
    augment_data(images_folder=images_folder,
                 extension=extension,
                 output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
