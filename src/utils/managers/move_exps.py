# move imgs module

# imports
from os import listdir
from os.path import join
from shutil import move as sh_move

# getting current folder
base_folder = ''
experiment = 'experimento14'
phase_folder_src = join(base_folder, 'move_folder', 'phase')
phase_folder_dst = join(base_folder, experiment, 'phase')
red_folder_src = join(base_folder, 'move_folder', 'red')
red_folder_dst = join(base_folder, experiment, 'red')

# defining keyword
keyword = ''

# getting files in input folder
all_files = listdir(phase_folder_src)
selected_files = [file for file in all_files if file.startswith(keyword)]

# moving images
for file in selected_files:
    print(file)
    continue
    phase_src = join(phase_folder_src, file)
    phase_dst = join(phase_folder_dst, file)
    file = file.replace('.jpg', '.tif')
    red_src = join(red_folder_src, file)
    red_dst = join(red_folder_dst, file)
    print(file)
    sh_move(phase_src, phase_dst)
    sh_move(red_src, red_dst)

# end of current module

