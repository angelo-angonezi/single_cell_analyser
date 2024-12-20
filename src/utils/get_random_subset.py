# imports
print('importing required libraries...')  # noqa
from os import listdir
from os.path import join
from os.path import exists
from pandas import read_csv
from shutil import copy as sh_cp
from src.utils.aux_funcs import print_progress_message
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

# defining global variables
seed = 53
base_folder = 'C:\\data_tmp\\angelo\\test'
df_path = join(base_folder, 'info_files', 'dataset_splits.csv')
red_folder_src = join(base_folder, 'imgs', 'red')
phase_folder_src = join(base_folder, 'imgs', 'phase')
annotation_folder_src = join(base_folder, 'annotations', 'rolabelimg_format', 'fornma')
red_folder_dst = join(base_folder, 'sample', 'imgs', 'red')
phase_folder_dst = join(base_folder, 'sample', 'imgs', 'phase')
annotation_folder_dst = join(base_folder, 'sample', 'annotations')

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

# defining placeholder for current index
current_index = 0

# iterating over groups
for df_name, df_group in df_groups:

    # updating current index
    current_index += 1

    # printing progress message
    base_string = 'copying file #INDEX# of #TOTAL#'
    print_progress_message(base_string=base_string,
                           index=current_index,
                           total=groups_num)

    # getting current group random image
    random_sample = df_group.sample(n=1,
                                    random_state=seed)

    # getting current group random image
    random_image = random_sample.iloc[0]['img_name']

    # getting file names
    tif_name = f'{random_image}.tif'
    jpg_name = f'{random_image}.jpg'
    xml_name = f'{random_image}.xml'

    # getting file src paths
    red_path_src = join(red_folder_src, tif_name)
    phase_path_src = join(phase_folder_src, jpg_name)
    annotation_path_src = join(annotation_folder_src, xml_name)

    # getting file dst paths
    red_path_dst = join(red_folder_dst, tif_name)
    phase_path_dst = join(phase_folder_dst, jpg_name)
    annotation_path_dst = join(annotation_folder_dst, jpg_name)

    # copying files
    sh_cp(src=red_path_src,
          dst=red_path_dst)
    sh_cp(src=phase_path_src,
          dst=phase_path_dst)
    sh_cp(src=annotation_path_src,
          dst=annotation_path_dst)

# end of current module
