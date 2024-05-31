# Общие библиотеки
import pandas as pd
import seaborn as sns
import numpy as np
import warnings

# Графика
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display

# Обработка данных
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Обучение
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer

# Модели
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Настройки
pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True
warnings.simplefilter(action='ignore', category=FutureWarning)

# Глобальная константа рандома
RNG = 77777


def print_corr_heatmap(df_corr):
    df_corr = np.abs(df_corr).replace(1, 0)

    mask = np.zeros_like(df_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(18, 7))

    sns.heatmap(df_corr, mask=mask, annot=True, fmt=".5f", linewidths=.1, cmap='coolwarm')
    plt.title('Матрица модулей коэффициентов корреляции', fontsize=15)
    plt.ylabel('Признак', fontsize=15)
    plt.xlabel('Признак', fontsize=15)


def describe_column_category(column, df, plot=False):
    print('Признак', column, ':\n')
    print('Уникальные значения (процент):')
    print(df[column].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

    if (plot):
        fig = px.histogram(
            df,
            x=column,
            opacity=0.7,
            title='Распределение значений признака {0}'.format(column)
        )

        fig.update_layout(xaxis_title='Значения признака {0}'.format(column), yaxis_title='Частота встречаемости')
        fig.show()


def describe_column_numeric(column, df):
    print('Признак', column, ':')

    print(df[column].describe())

    fig = px.histogram(
        df,
        x=column,
        marginal='box',
        opacity=0.7,
        title='Распределение значений признака {0}'.format(column)
    )

    fig.update_layout(xaxis_title='Значения признака {0}'.format(column), yaxis_title='Частота встречаемости')
    fig.show()


def print_corr_data(df, corr_matrix=True):
    df_corr = df.corr()
    display(df_corr)
    print_corr_heatmap(df_corr)
    if corr_matrix:
        pd.plotting.scatter_matrix(
            df,
            figsize=(12, 12)
        )


main_df = pd.read_csv("Housing_v2.csv", index_col='Id')

columns = ['ms_sub_class', 'ms_zoning', 'lot_area', 'lot_config', 'bldg_type',
           'overall_cond', 'year_built', 'year_remod_add', 'exterior_1st', 'bsmt_fin_sf2',
           'total_bsmt_sf', 'sale_price']

main_df.columns = columns

data = main_df[main_df['sale_price'].notna()]

categorial = ['ms_zoning', 'lot_config', 'bldg_type', 'exterior_1st']

data = data[(data['ms_zoning'] != 'C (all)') & (data['ms_zoning'] != 'RH')]

data = data[data['lot_config'] != 'FR3']

to_drop = ['BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock']
data = data[(data["exterior_1st"] != 'BrkComm') & (data['exterior_1st'] != 'Stone')]
data = data[(data["exterior_1st"] != 'AsphShn') & (data['exterior_1st'] != 'ImStucc')]
data = data[data["exterior_1st"] != 'CBlock']

numeric = ['ms_sub_class', 'lot_area', 'overall_cond', 'year_built', 'year_remod_add', 'bsmt_fin_sf2', 'total_bsmt_sf',
           'sale_price']

data = data[data['lot_area'] < 45000]

data = data[data['overall_cond'] > 3]

data = data[data['year_built'] >= 1885]

data = data[(data['total_bsmt_sf'] > 105) & (data['total_bsmt_sf'] < 2006)]

data = data[data['sale_price'] < 500000]

target = data['sale_price']
features = data.drop('sale_price', axis=1)

std_scaler = StandardScaler()
pipe_num = Pipeline([('scaler', std_scaler)])

ohe_encoder = OneHotEncoder(handle_unknown='ignore')
pipe_cat = Pipeline([('encoder', ohe_encoder)])

col_transformer = ColumnTransformer(
    [('num_preproc', pipe_num, [x for x in features.columns if features[x].dtype != 'object']),
     ('cat_preproc', pipe_cat, [x for x in features.columns if features[x].dtype == 'object'])])

res = col_transformer.fit_transform(features)
res_df = pd.DataFrame(res, columns=[x.split('__')[-1] for x in col_transformer.get_feature_names_out()])

features_train, features_valid, target_train, target_valid = train_test_split(res_df, target,
                                                                              train_size=0.75, random_state=RNG)

model_linear = LinearRegression()

mse = make_scorer(mean_squared_error)

cross_scores = cross_val_score(model_linear, features_train, target_train, scoring=mse, cv=5)

cv_results = cross_validate(model_linear, features_train, target_train, scoring=mse, cv=5, return_estimator=True)


def showcase():
    while True:
        command = input()
        if command == 'Show graphics':
            print_corr_data(data[numeric])

        elif command == 'Show categorial':
            describe_column_category('ms_zoning', data, True)
            describe_column_category('lot_config', data, True)
            describe_column_category('bldg_type', data, True)
            describe_column_category('exterior_1st', data, True)
        elif command == 'Show numeric':
            describe_column_numeric('ms_sub_class', data)
            describe_column_numeric('lot_area', data)
            describe_column_category('overall_cond', data, True)
            describe_column_numeric('year_built', data)
            describe_column_numeric('year_remod_add', data)
            describe_column_numeric('bsmt_fin_sf2', data)
            describe_column_numeric('total_bsmt_sf', data)
            describe_column_numeric('sale_price', data)
        elif command == 'Exit':
            break
        elif command == 'Price median':
            target_train.median()


showcase()
