# imports
from os import listdir
from pandas import read_csv
from os.path import join
from os.path import exists
from shutil import copy as sh_cp
from random import randint
# defining global variables
seed = randint(0, 10000)
base_folder = 'C:\\data_tmp\\angelo\\test'
df_path = join(base_folder, 'info_files', 'dataset_splits.csv')
red_folder = join(base_folder, 'imgs', 'red')
phase_folder = join(base_folder, 'imgs', 'phase')
red_files = listdir(red_folder)
phase_file = listdir(phase_folder)

# reading df
df = read_csv(df_path)

# filtering df
df = df[df['split'] == 'test']

# defining groups
groups_list = ['cell_line',
               'treatment',
               'confluence_group']

# grouping df
df_groups = df.groupby(groups_list)

# getting groups num
groups_num = len(df_groups)

# printing execution message
f_string = f'{groups_num} groups were found based on: {groups_list}'
print(f_string)

# defining placeholder value for files list
random_files = []
random_red = []
random_phase = []

# iterating over groups
for df_name, df_group in df_groups:

    # starting loop
    while True:

        # getting current group random image
        random_sample = df_group.sample(n=1,
                                        random_state=seed)

        # getting current group random image
        random_image = random_sample.iloc[0]['img_name']

        # appending current image to files list
        random_files.append(random_image)

        # getting file names
        tif_name = f'{random_image}.tif'
        jpg_name = f'{random_image}.jpg'

        # getting file paths
        red_path = join(red_folder, tif_name)
        phase_path = join(phase_folder, jpg_name)

        # checking wheteher files exists
        if exists(red_path):
            random_red.append(red_path)
        else:
            print('Red file does not exist!')
            continue
        if exists(phase_path):
            random_phase.append(phase_path)
        else:
            print('Phase file does not exist!')

print('seed: ', seed)
print(len(random_red))
print(len(random_phase))


















