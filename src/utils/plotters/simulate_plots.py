# imports
from random import uniform
from seaborn import boxplot
from pandas import DataFrame
from matplotlib import pyplot as plt

# defining parameters
min_value = 0.92
max_value = 0.62
elements_num = 10
elements_range = range(elements_num)

# defining simulated col values
U87_col = []
U251_col = []
A172_col = []
MRC5_col = []
for i in elements_range:
    U87_col.append(uniform(min_value, max_value))
    U251_col.append(uniform(min_value, max_value))
    A172_col.append(uniform(min_value, max_value))
    MRC5_col.append(uniform(min_value, max_value))


# assembling data dict
data_dict = {'U87': U87_col,
             'U251': U251_col,
             'A172': A172_col,
             'MRC5': MRC5_col}

conf_data_dict = {'Low': [uniform(0.51, 0.75) for _ in elements_range],
                  'High': [uniform(0.32, 0.41) for _ in elements_range]}

# converting data to data frame
df = DataFrame(conf_data_dict)

# melting df
melted_df = df.melt()

# plotting data
boxplot(data=melted_df,
        x='variable',
        y='value')

# setting axis labels
x_label = 'Confluence'
y_label = 'F1-Score'
plt.xlabel(x_label)
plt.ylabel(y_label)

# showing plot
plt.show()

# end of current module
