# imports
from os import listdir
from os.path import join
from os.path import exists
from pandas import read_csv
from shutil import copy as sh_cp

# defining global variables
seed = 53
base_folder = 'C:\\data_tmp\\angelo\\test'
df_path = join(base_folder, 'info_files', 'dataset_splits.csv')
red_folder = join(base_folder, 'imgs', 'red')
phase_folder = join(base_folder, 'imgs', 'phase')
annotation_folder = join(base_folder, 'annotations', 'rolabelimg_format', 'fornma')

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

# iterating over groups
for df_name, df_group in df_groups:

    # getting current group random image
    random_sample = df_group.sample(n=1,
                                    random_state=seed)

    # getting current group random image
    random_image = random_sample.iloc[0]['img_name']

    # getting file names
    tif_name = f'{random_image}.tif'
    jpg_name = f'{random_image}.jpg'
    xml_name = f'{random_image}.xml'

    # getting file paths
    red_path = join(red_folder, tif_name)
    phase_path = join(phase_folder, jpg_name)
    annotation_path = join(annotation_folder, jpg_name)

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


















