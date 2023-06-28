# imports
from os import listdir
from os.path import join
from shutil import copy as sh_copy

# defining global variables
BASE_FOLDER = '/home/angelo/dados/pycharm_projects/single_cell_analyser/data/ml/nucleus_detection/NucleusDetectorMerge'

FORNMA_OUTPUT_FOLDER = join(BASE_FOLDER,
                            'fornma_output')

RED = join(BASE_FOLDER, 'imgs', 'red')
RED_FORNMA_FOUND = join(BASE_FOLDER, 'imgs', 'red_fornma_found')
RED_FORNMA_MISSED = join(BASE_FOLDER, 'imgs', 'red_fornma_missed')

PHASE = join(BASE_FOLDER, 'imgs', 'phase')
PHASE_FORNMA_FOUND = join(BASE_FOLDER, 'imgs', 'phase_fornma_found')
PHASE_FORNMA_MISSED = join(BASE_FOLDER, 'imgs', 'phase_fornma_missed')

# getting images found by forNMA
FORNMA_FOUND_FILES = listdir(FORNMA_OUTPUT_FOLDER)
FORNMA_FOUND_FILES = [f.replace('.csv', '.tif') for f in FORNMA_FOUND_FILES]
ALL_FILES = listdir(RED)
ALL_FILES_NUM = len(ALL_FILES)

# iterating over files in input folder
for file_index, file in enumerate(ALL_FILES, 1):

    # defining progress string
    p_string = f'copying file {file_index} of {ALL_FILES_NUM} '

    # checking if file is in output folder
    if file in FORNMA_FOUND_FILES:

        # updating progress string
        p_string += '(nuclei found)'

        # if it is, move files to FOUND path
        old_red_file_path = join(RED, file)
        new_red_file_path = join(RED_FORNMA_FOUND, file)

        # adapting file format for phase images
        file = file.replace('.tif', '.jpg')
        old_phase_file_path = join(PHASE, file)
        new_phase_file_path = join(PHASE_FORNMA_FOUND, file)

    else:

        # updating progress string
        p_string += '(no nuclei found)'

        # if it is not, move files to MISSED path
        old_red_file_path = join(RED, file)
        new_red_file_path = join(RED_FORNMA_MISSED, file)

        # adapting file format for phase images
        file = file.replace('.tif', '.jpg')
        old_phase_file_path = join(PHASE, file)
        new_phase_file_path = join(PHASE_FORNMA_MISSED, file)

    # printing progress string
    print(p_string)

    # moving files
    sh_copy(old_red_file_path, new_red_file_path)
    sh_copy(old_phase_file_path, new_phase_file_path)

# end of current module

