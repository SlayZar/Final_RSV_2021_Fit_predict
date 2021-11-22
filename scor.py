import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import base64
import time
import plotly
import plotly_express as px
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch
from sklearn.manifold import TSNE
import streamlit.components.v1 as components
from matplotlib import pyplot as pl

import shap
import lightgbm as lgb
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Скачать результаты в формате csv</a>'
    return href

def slider_feats(train, feat, target_col_name):
    try:
        grouped = train.groupby(feat)[target_col_name].mean().to_frame(target_col_name).sort_values(by=target_col_name, ascending=False)
        values = list(grouped.index) # 
    except:
        values = sorted(train[feat].unique())
    
    st.write('Результат работы модели для различных ID')
    ids = st.selectbox(f'Выберите {feat}', values)
    df_samp = train[train['id']==ids].copy()
    df_samp.set_index('time', inplace=True)
    df_samp['Аномалия ритма сердца'] = df_samp['x'] * df_samp[target_col_name].replace(0, np.nan)
    try:
        st.line_chart(df_samp[['x', 'Аномалия ритма сердца']])
    except:
        pass

    
def visualisation(df, numeric_columns, non_numeric_columns):
    chart_select = st.sidebar.selectbox(
        label="Выберите тип графиков",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )

    if chart_select == 'Scatterplots':
        st.sidebar.subheader("Scatterplot")
        x_values = st.sidebar.selectbox('Ось X', options=numeric_columns)
        y_values = st.sidebar.selectbox('Ось Y', options=numeric_columns)
        color_value = st.sidebar.selectbox("Цвет", options=non_numeric_columns)
        plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
        # display the chart
        st.plotly_chart(plot)
    if chart_select == 'Lineplots':
        st.sidebar.subheader("Линейные графики")
        x_values = st.sidebar.selectbox('Ось X', options=numeric_columns)
        y_values = st.sidebar.selectbox('Ось Y', options=numeric_columns)
        color_value = st.sidebar.selectbox("Цвет", options=non_numeric_columns)
        plot = px.line(data_frame=df.sort_values(by=x_values), x=x_values, y=y_values, color=color_value)
        st.plotly_chart(plot)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Гистограмма")
        x = st.sidebar.selectbox('Признаки', options=numeric_columns)
        bin_size = st.sidebar.slider("Число бинов", min_value=10,
                                         max_value=100, value=40)
        color_value = st.sidebar.selectbox("Цвет", options=non_numeric_columns)
        plot = px.histogram(x=x, data_frame=df, color=color_value, nbins=bin_size)
        st.plotly_chart(plot)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot")
        y = st.sidebar.selectbox("Ось Y", options=numeric_columns)
        x = st.sidebar.selectbox("Ось X", options=non_numeric_columns)
        color_value = st.sidebar.selectbox("Цвет", options=non_numeric_columns)
        plot = px.box(data_frame=df, y=y, x=x, color=color_value)
        st.plotly_chart(plot)
    
# if st.sidebar.button('Очистить кэш'):
#     st.caching.clear_cache()

# data_path = 'data/test.csv'
# target_col_name = 'pred2_bin'

def preproc(customers):
    df = pd.pivot_table(customers, index = ['CustomerID'], columns = ['Gender'], 
                    values = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
                    aggfunc = 'mean').fillna(0)
    df.columns = ['Female_age', 'Male_age', 'Female_income', 'Male_Income', 'Female_score', 'Male_score']
    return df


def interpret(df):
    from config import MODEL_COLS, MODEL_PATH, TARGET_COL
    df = preproc(df)
#     st.write(df[MODEL_COLS])
    clf = joblib.load('models/clf') #Закомментить
    df[TARGET_COL] = clf.predict(df[MODEL_COLS]) #Закомментить
    loaded_model_cb = CatBoostClassifier()
    loaded_model_cb.load_model('models/cb') 
#     loaded_model = XGBClassifier()
#     loaded_model.load_model(MODEL_PATH) 
    gbm = joblib.load('models/lightgbm')
    lgb_train = lgb.Dataset(df[MODEL_COLS], df[TARGET_COL])

    
    ntree=st.number_input('Select the desired record for detailed explanation on the training set'
                                       , min_value=0
                                       , max_value=df.shape[0]
                                       )
    graph = lgb.create_tree_digraph(gbm, tree_index=ntree, name='Tree54')
    st.graphviz_chart(graph)
#     tree=xgb.to_graphviz(loaded_model,num_trees=ntree)
#     st.graphviz_chart(tree)
#     importance_type = st.selectbox('Select the desired importance type', ('weight','gain','cover'),index=0)
#     importance_plot = xgb.plot_importance(loaded_model,importance_type=importance_type)
#     pl.title ('xgboost.plot_importance(best XGBoost model) importance type = '+ str(importance_type))
#     st.pyplot(bbox_inches='tight')
#     pl.clf()
#     st.write('To handle this inconsitency, SHAP values give robust details, among which is feature importance')
    explainer = shap.TreeExplainer(loaded_model_cb)
    shap_values = explainer.shap_values(df[MODEL_COLS])
    pl.title('Assessing feature importance based on Shap values')
    shap.summary_plot(shap_values,df[MODEL_COLS],plot_type="bar",show=False)
    st.pyplot(bbox_inches='tight')
    pl.clf()
    expectation = explainer.expected_value
    individual = st.number_input('Select the desired record from the training set for detailed explanation.'
                                            , min_value=0
                                       , max_value=df.shape[0]
                                       )
    predicted_values = loaded_model_cb.predict(df[MODEL_COLS])
    real_value = df[TARGET_COL]    
    shap.force_plot(explainer.expected_value[0], shap_values[0][individual], # Убрать нолики
                    df[MODEL_COLS].iloc[individual,:],matplotlib=True,show=False
                    ,figsize=(16,5))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()
    if st.button('Click here to see a drilldown of the SHAP values'):
        shap_table=pd.DataFrame(shap_values,columns=MODEL_COLS)
        st.table(shap_table.iloc[individual])

def eda_func(df):
    if st.checkbox("Показать 10 строк датафрейма"):
        st.write(df.head(10))           

    st.sidebar.subheader("Настройки визуализации")
    dropcols = ['CustomerID']
    numeric_columns = list(set(df.select_dtypes(['float', 'int64']).columns).difference(dropcols))
    non_numeric_columns = list(set(df.select_dtypes(['object']).columns).difference(dropcols))
    non_numeric_columns.append(None)
    options2 = st.selectbox('Что хотим сделать?',
             ['Посмотреть зависимости (EDA)', 'Кластеризация', 'Интерпретация модели'], index=0)
    if options2 == 'Посмотреть зависимости (EDA)':
        visualisation(df, numeric_columns, non_numeric_columns)
    elif options2 == 'Кластеризация': 
        hierarch(df)
    elif options2 == 'Интерпретация модели': 
        interpret(df)
        
def hierarch(customers):
    df = preproc(df)
    X_embedded = TSNE(n_components=2, perplexity = 5, verbose = 0).fit_transform(df.to_numpy())
    vis_df = pd.DataFrame(X_embedded, index = df.index)
    fig = ff.create_dendrogram(np.array(vis_df[[0,1]]),
                               orientation='left',
                               linkagefun=lambda x: sch.linkage(np.array(vis_df[[0,1]]), "average"),)
    fig.update_layout(width=800, height=1600, font_size=8)
    # fig.show()
    # plotly.offline.plot(fig, filename='cluster.html')
    HtmlFile = open("cluster.html", 'r', encoding='utf-8')
    components.html(HtmlFile.read(), width=1000, height=5000, scrolling=True)
       
st.set_page_config("Fit_Predict Final case solution")
st.image("https://i.ibb.co/Vwhhs7J/image.png", width=150)

# Title of the main page
st.title("Профилирование клиентов")   
     
options = st.selectbox('Какие данные анализируем?',
         ['Тестовый датасет', 'Загрузить новый датасет'], index=0)


# Считывание датафрейма
if options == 'Тестовый датасет':
    df = pd.read_csv('data/Mall_Customers.csv')
    eda_func(df)
else:
    file_buffer = st.file_uploader(label = 'Выберите датасет')
    if file_buffer:
        try:
            df = pd.read_csv(file_buffer, encoding=None)
            eda_func(df)
        except:
            st.write('Файл некорректен!')
    

