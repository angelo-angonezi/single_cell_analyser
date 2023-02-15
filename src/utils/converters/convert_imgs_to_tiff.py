# convert images to tiff format module

######################################################################
# imports

# importing required libraries
from PIL import Image
from os import listdir
from os.path import join

######################################################################
# defining global variables

INPUT_DIR = 'orig'
OUTPUT_DIR = 'tiff_format'

######################################################################
# running code

# getting images in input dir

files_in_dir = listdir(INPUT_DIR)

# iterating over files

for file in files_in_dir:

    # printing execution message
    f_string = f'converting image "{file}"'
    print(f_string)

    # getting file path
    file_path = join(INPUT_DIR, file)

    # opening image
    img = Image.open(file_path)

    # saving image in output folder as tiff
    save_name = file.replace('.png', '.tif')
    save_name = save_name.replace('.jpeg', '.tif')
    save_name = save_name.replace('.jpg', '.tif')
    save_path = join(OUTPUT_DIR, save_name)
    img.save(save_path, 'TIFF')

# printing execution message
f_string = f'All images converted!'
print(f_string)

######################################################################
# end of current module
