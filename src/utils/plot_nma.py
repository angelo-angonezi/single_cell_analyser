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
                '\\fornma_output'
                '\\Results.csv')
CELL_CYCLE_INPUT = ('Z:'
                    '\\pycharm_projects'
                    '\\single_cell_analyser'
                    '\\data'
                    '\\cell_cycle_inference'
                    '\\debs'
                    '\\sic'
                    '\\plots'
                    '\\histograms'
                    '\\cell_cycle_df.csv')
OUTPUT_FOLDER = ('Z:'
                 '\\pycharm_projects'
                 '\\single_cell_analyser'
                 '\\data'
                 '\\cell_cycle_inference'
                 '\\debs'
                 '\\sic'
                 '\\output')

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
desired_cols = ['Area', 'NII', 'CellCycle', 'Month', 'Treatment', 'Datetime']

from src.utils.aux_funcs import drop_unrequired_cols
drop_unrequired_cols(df=fornma_df,
                     cols_to_keep=desired_cols)

print(fornma_df[fornma_df['Treatment'] == 'CTR'])
a = '2023y05m31d17h30m'
b = '2023y06m01d17h30m'
c = '2023y06m02d23h30m'
d = [a, b, c]
fornma_df = fornma_df[fornma_df['Datetime'].isin(d)]
print(fornma_df)

from os.path import join
# grouping df
treatment_groups = fornma_df.groupby('Treatment')
for treatment, treatment_group in treatment_groups:

    sns.scatterplot(data=treatment_group,
                    x='NII',
                    y='Area',
                    hue='CellCycle',
                    hue_order=['G1', 'S', 'G2', 'M'],
                    palette=['red', 'yellow', 'green', 'gray'])
    title = f'Fucci-NMA plot (T: {treatment})'
    save_name = f'{treatment}_plot.png'
    plt.title(title)
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

plt.show()


# end of the current module
