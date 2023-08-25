# NMA plots module

# import
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

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

# functions
# le csvs
fornma_df = pd.read_csv(FORNMA_INPUT)
cell_cycle_df = pd.read_csv(CELL_CYCLE_INPUT)

print(fornma_df)
print(cell_cycle_df)
cell_cycle_col = cell_cycle_df['cell_cycle']

# adding cell cycle col to fornma df
fornma_df['CellCycle'] = cell_cycle_col
print(fornma_df)

# def colunas q tu quer e armazena elas numa lista
desired_cols = ['Area', 'NII', 'CellCycle']

from src.utils.aux_funcs import drop_unrequired_cols
drop_unrequired_cols(df=fornma_df,
                     cols_to_keep=desired_cols)

print(fornma_df)

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
