# NMA plots module

# import
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from src.utils.aux_funcs import get_analysis_df

# variaveis globais
FORNMA_INPUT = ('Z:'
                '\\pycharm_projects'
                '\\single_cell_analyser'
                '\\data'
                '\\cell_cycle_inference'
                '\\debs'
                '\\sic'
                '\\fornma_output2'
                '\\Results.csv')
CELL_CYCLE_INPUT = ('Z:'
                    '\\pycharm_projects'
                    '\\single_cell_analyser'
                    '\\data'
                    '\\cell_cycle_inference'
                    '\\debs'
                    '\\sic'
                    '\\plots'
                    '\\histograms2'
                    '\\cell_cycle_df.csv')
OUTPUT_FOLDER = ('Z:'
                 '\\pycharm_projects'
                 '\\single_cell_analyser'
                 '\\data'
                 '\\cell_cycle_inference'
                 '\\debs'
                 '\\sic'
                 '\\output2')

TREATMENT_DICT = {'A2': 'CTR',
                  'B2': 'TMZ'}

# functions
# le csvs
fornma_df = get_analysis_df(fornma_file_path=FORNMA_INPUT,
                            image_name_col='Image_name_rg_merge',
                            output_folder=OUTPUT_FOLDER,
                            treatment_dict=TREATMENT_DICT)
cell_cycle_df = pd.read_csv(CELL_CYCLE_INPUT)

print(fornma_df)
print(cell_cycle_df)
cell_cycle_col = cell_cycle_df['cell_cycle']

# adding cell cycle col to fornma df
fornma_df['CellCycle'] = cell_cycle_col
print(fornma_df)

# def colunas q tu quer e armazena elas numa lista
desired_cols = ['Image_name_rg_merge', 'Area', 'NII', 'CellCycle', 'Month', 'Treatment', 'Datetime']

from src.utils.aux_funcs import drop_unrequired_cols
drop_unrequired_cols(df=fornma_df,
                     cols_to_keep=desired_cols)

from os.path import join
# grouping df
treatment_groups = fornma_df.groupby('Treatment')
for treatment, treatment_group in treatment_groups:

    print(treatment)
    print(treatment_group['Image_name_rg_merge'].unique())
    print(len(treatment_group['Image_name_rg_merge'].unique()))

    sns.scatterplot(data=treatment_group,
                    x='NII',
                    y='Area',
                    hue='CellCycle',
                    hue_order=['G1', 'S', 'G2', 'M'],
                    palette=['red', 'yellow', 'green', 'gray'])

    # getting means
    g1_group = treatment_group[treatment_group['CellCycle'] == 'G1']
    g2_group = treatment_group[treatment_group['CellCycle'] == 'G2']
    g1_area_col = g1_group['Area']
    g2_area_col = g2_group['Area']
    g1_area_mean = g1_area_col.mean()
    g2_area_mean = g2_area_col.mean()

    # adding means to plot
    plt.axhline(y=g1_area_mean, c='red', linestyle='--')
    plt.axhline(y=g2_area_mean, c='green', linestyle='--')

    title = f'Fucci-NMA plot (T: {treatment})'
    save_name = f'{treatment}_plot.png'
    plt.title(title)
    plt.ylim(0, 3800)
    plt.xlim(2, 10)
    output_path = join(OUTPUT_FOLDER, save_name)
    plt.savefig(output_path)
    plt.close()

# .qqer coisa é um atributo, faz p pegar adj de coisas
cols = fornma_df.columns
'''
# sim loop de for em py eh idiota
for col in cols:
    # usar o in nesse caso faz com que tu possa pegar qualquer elemento da lista, ao inves d comparar a lista inteira
    if col in desired_cols:
        print(col)
'''
# plot do gráfico
sns.scatterplot(data=fornma_df,
                x='NII',
                y='Area',
                hue='CellCycle',
                hue_order=['G1', 'S', 'G2', 'M'],
                palette=['red', 'yellow', 'green', 'gray'])

# setting configs do plot
# opção de fazer globalmente com sns.set_context() ou sns.set_theme()
# mas testei e ficou ruim
plt.title('Fucci-NMA',
          fontsize=12)
plt.xlabel('NII',
           fontsize=12)
plt.ylabel('Area',
           fontsize=12)

# plt.show()

print('done!')

# end of the current module
