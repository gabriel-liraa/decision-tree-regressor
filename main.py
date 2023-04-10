##
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

##


def trainModel(X, y, nodes):
    model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=nodes)
    model.fit(X, y)
    return model


##


def fittingViz(X, y, cont):
    nodes = list(map(int, np.linspace(10, y.shape[0], cont).round()))
    plot_y = []
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    for c in nodes:
        model = trainModel(train_X, train_y, c)
        pred = model.predict(val_X)
        plot_y.append(mean_absolute_error(val_y, pred))
    best_fit = plot_y.index(min(plot_y))
    plot = sns.lineplot(x=nodes, y=plot_y)
    plot.set(title=f'Best fit {nodes[best_fit]}', xlabel='nodes', ylabel='MAE')


##

df = pd.read_csv('melb_data.csv')

df = df.dropna(axis=0)

##

y = df['Price']

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = df[features]

##

fittingViz(X, y, 150)

##

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model_none = trainModel(train_X, train_y, None)
pred_none = model_none.predict(val_X)
mae_none = mean_absolute_error(val_y, pred_none)

model_799 = trainModel(train_X, train_y, 799)
pred_799 = model_799.predict(val_X)
mae_799 = mean_absolute_error(val_y, pred_799)

##

df_comp = pd.DataFrame({'MAE': [mae_799, mae_none]}, index=['max_799', 'None'])

df_comp

##

df_plot_y = pd.DataFrame({'data': val_y,
                          'type': ['validation' for i in range(val_y.shape[0])]}
                         ).reset_index(drop=True)


df_plot_799 = pd.DataFrame({'data': pred_799,
                            'type': ['optimized' for i in range(val_y.shape[0])]}
                           ).reset_index(drop=True)

df_plot_y_799 = pd.concat([df_plot_y, df_plot_799])

sns.scatterplot(data=df_plot_y_799.reset_index(),
                x='index',
                y='data',
                hue='type',
                marker='+').set(title=f'MAE={mae_799}',
                                ylabel='Price')

##

df_plot_none = pd.DataFrame({'data': pred_none,
                            'type': ['standard' for i in range(val_y.shape[0])]}
                            ).reset_index(drop=True)

df_plot_y_none = pd.concat([df_plot_y, df_plot_none])

sns.scatterplot(data=df_plot_y_none.reset_index(),
                x='index',
                y='data',
                hue='type',
                marker='+').set(title=f'MAE={mae_none}',
                                ylabel='Price')

##
