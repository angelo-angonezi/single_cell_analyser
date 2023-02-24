# imports
from os import rename
from os import listdir
from os.path import join
from shutil import copy as sh_copy

# defining global variables
FORNMA_OUTPUT_FOLDER = '/home/angelo/dados/ml_imgs/TreinoMLIncucyteNucleus/fornma_output/red/53bp1'

RED = '/home/angelo/dados/ml_imgs/TreinoMLIncucyteNucleus/imgs/red'
RED_FORNMA_FOUND = '/home/angelo/dados/ml_imgs/TreinoMLIncucyteNucleus/imgs/red_fornma_found'
RED_FORNMA_MISSED = '/home/angelo/dados/ml_imgs/TreinoMLIncucyteNucleus/imgs/red_fornma_missed'

PHASE = '/home/angelo/dados/ml_imgs/TreinoMLIncucyteNucleus/imgs/phase'
PHASE_FORNMA_FOUND = '/home/angelo/dados/ml_imgs/TreinoMLIncucyteNucleus/imgs/phase_fornma_found'
PHASE_FORNMA_MISSED = '/home/angelo/dados/ml_imgs/TreinoMLIncucyteNucleus/imgs/phase_fornma_missed'

# getting images found by forNMA
FORNMA_FOUND_FILES = listdir(FORNMA_OUTPUT_FOLDER)
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

        old_phase_file_path = join(PHASE, file)
        new_phase_file_path = join(PHASE_FORNMA_FOUND, file)

    else:

        # updating progress string
        p_string += '(no nuclei found)'

        # if it is not, move files to MISSED path
        old_red_file_path = join(RED, file)
        new_red_file_path = join(RED_FORNMA_MISSED, file)

        old_phase_file_path = join(PHASE, file)
        new_phase_file_path = join(PHASE_FORNMA_MISSED, file)

    # printing progress string
    print(p_string)

    # moving files
    sh_copy(old_red_file_path, new_red_file_path)
    sh_copy(old_phase_file_path, new_phase_file_path)

# end of current module

