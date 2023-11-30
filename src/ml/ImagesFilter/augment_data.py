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
from cv2 import convertScaleAbs
from argparse import ArgumentParser
from src.utils.aux_funcs import IMAGE_SIZE
from src.utils.aux_funcs import resize_image
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

    # resize param
    parser.add_argument('-r', '--resize',
                        dest='resize',
                        action='store_true',
                        required=False,
                        default=False,
                        help='defines whether or not to resize images.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def augment_image(image_name: str,
                  images_folder: str,
                  output_folder: str,
                  resize: bool
                  ) -> None:
    # getting current image path
    image_path = join(images_folder,
                      image_name)

    # opening current image
    open_image = imread(image_path)

    # defining save path
    save_path = join(output_folder,
                     image_name)

    # checking resize toggle
    if resize:

        # resizing image
        open_image = resize_image(open_image=open_image,
                                  image_size=IMAGE_SIZE)

    # getting rotated image
    rotated_image = rotate(open_image,
                           ROTATE_180)

    # getting vertically flipped image
    v_flipped_image = flip(open_image,
                           0)

    # getting horizontally flipped image
    h_flipped_image = flip(open_image,
                           1)

    # defining alpha and beta
    alpha_d = 0.9  # Contrast control
    beta_d = -2  # Brightness control
    alpha_u = 1.1  # Contrast control
    beta_u = 5  # Brightness control

    # getting contrast/brightness changed image
    od_contrast_image = convertScaleAbs(open_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    rd_contrast_image = convertScaleAbs(rotated_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    vd_contrast_image = convertScaleAbs(v_flipped_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    hd_contrast_image = convertScaleAbs(h_flipped_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    ou_contrast_image = convertScaleAbs(open_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    ru_contrast_image = convertScaleAbs(rotated_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    vu_contrast_image = convertScaleAbs(v_flipped_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    hu_contrast_image = convertScaleAbs(h_flipped_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    # saving images
    imwrite(save_path.replace('.jpg', '_o.jpg'), open_image)
    imwrite(save_path.replace('.jpg', '_r.jpg'), rotated_image)
    imwrite(save_path.replace('.jpg', '_v.jpg'), v_flipped_image)
    imwrite(save_path.replace('.jpg', '_h.jpg'), h_flipped_image)
    imwrite(save_path.replace('.jpg', '_od.jpg'), od_contrast_image)
    imwrite(save_path.replace('.jpg', '_rd.jpg'), rd_contrast_image)
    imwrite(save_path.replace('.jpg', '_vd.jpg'), vd_contrast_image)
    imwrite(save_path.replace('.jpg', '_hd.jpg'), hd_contrast_image)
    imwrite(save_path.replace('.jpg', '_ou.jpg'), ou_contrast_image)
    imwrite(save_path.replace('.jpg', '_ru.jpg'), ru_contrast_image)
    imwrite(save_path.replace('.jpg', '_vu.jpg'), vu_contrast_image)
    imwrite(save_path.replace('.jpg', '_hu.jpg'), hu_contrast_image)


def augment_data(images_folder: str,
                 extension: str,
                 output_folder: str,
                 resize: bool
                 ) -> None:
    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=images_folder,
                                          extension=extension)

    # getting images num
    images_num = len(images)

    # setting number of modifications
    mods_num = 12  # imwrite calls count inside augment_image
    final_imgs_num = images_num * mods_num

    # printing execution message
    f_string = f'found {images_num} images in input folder.'
    print(f_string)

    # defining placeholder value for current_image_index
    current_image_index = 1

    # iterating over images
    for image in images:

        # getting current augmented images
        current_augmented_images = current_image_index * mods_num

        # printing execution message
        base_string = f'augmenting image #INDEX# of #TOTAL# (total imgs: {current_augmented_images})'
        print_progress_message(base_string=base_string,
                               index=current_image_index,
                               total=images_num)

        # augmenting current image
        augment_image(image_name=image,
                      images_folder=images_folder,
                      output_folder=output_folder,
                      resize=resize)

        # updating current_image_index
        current_image_index += 1

    # printing execution message
    print('augmentation complete!')
    print(f'augmented data set now contains {final_imgs_num} images.')
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

    # getting resize param
    resize = bool(args_dict['resize'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running augment_data function
    augment_data(images_folder=images_folder,
                 extension=extension,
                 output_folder=output_folder,
                 resize=resize)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
