import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import base64
import time
from hrvanalysis import remove_outliers, remove_ectopic_beats
import plotly_express as px
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch
from sklearn.manifold import TSNE


# @st.cache(suppress_st_warning=True)
# def scoring(test_df, path_to_model, new_cols2, tresh, target_col='pred2_bin'):
#     test_df = anomaly_detected(test_df)
#     test_df = first_prepr(test_df, delete_anomaly = False)
#     df_test = test_df[['id', 'time', 'x']].copy()

#     test_df = feature_generate(test_df)
#     pred_df = test_df[['id', 'time', 'x', 'anomaly__1', 'less066','mean_rolling_new2_1']].copy()

#     pred_df = pred_df[(pred_df.anomaly__1==1) |(pred_df.less066==1)|(pred_df.mean_rolling_new2_1==1)]
#     pred_df['pred'] = 0
#     test_df = test_df[(test_df.anomaly__1==0) &(test_df.less066==0)&(test_df.mean_rolling_new2_1==0)]
#     cb = CatBoostClassifier()
#     cb.load_model(path_to_model)
#     test_df['pred2'] = cb.predict_proba(Pool(test_df[new_cols2]))[:,1]
#     test_df[target_col] = (test_df['pred2'] > tresh).astype(int)

#     df_test = df_test.merge(pred_df[['id', 'time', 'x', 'pred']], 
#                           on =['id', 'time', 'x'], how='left')
#     df_test = df_test.merge(test_df[['id', 'time', 'x', 'pred2', 'pred2_bin']], 
#                           on =['id', 'time', 'x'], how='left')
#     df_test.loc[(df_test.pred2_bin.isnull()), target_col] = df_test.loc[(df_test[target_col].isnull()), 'pred']
#     return df_test

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
        label="Select the chart type",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )

    if chart_select == 'Scatterplots':
        st.sidebar.subheader("Scatterplot Settings")
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
        plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
        # display the chart
        st.plotly_chart(plot)
    if chart_select == 'Lineplots':
        st.sidebar.subheader("Line Plot Settings")
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
        plot = px.line(data_frame=df.sort_values(by=x_values), x=x_values, y=y_values, color=color_value)
        st.plotly_chart(plot)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        x = st.sidebar.selectbox('Feature', options=numeric_columns)
        bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                         max_value=100, value=40)
        color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
        plot = px.histogram(x=x, data_frame=df, color=color_value, nbins=bin_size)
        st.plotly_chart(plot)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        y = st.sidebar.selectbox("Y axis", options=numeric_columns)
        x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
        color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
        plot = px.box(data_frame=df, y=y, x=x, color=color_value)
        st.plotly_chart(plot)

# def slider_to_cat(train, feat):
#     try:
#         values = sorted(train[feat].unique())
#     except:
#         values = train[feat].unique()
#     sexs = st.sidebar.selectbox(
#            f'{feat}', values)
#     return sexs

# def slider_to_non_cat(train, feat):
#     ages = st.sidebar.slider(feat, np.floor(train[feat].min()), np.ceil(train[feat].max()), train[feat].median())
#     return ages
 
    
# if st.sidebar.button('Очистить кэш'):
#     st.caching.clear_cache()

# tresh = 0.265
# data_path = 'data/test.csv'
# target_col_name = 'pred2_bin'
def eda_func(df):
    if st.checkbox("Show dataframe"):
        st.write(df.head(10))           

    st.sidebar.subheader("Visualization Settings")
    dropcols = ['CustomerID']
    numeric_columns = list(set(df.select_dtypes(['float', 'int64']).columns).difference(dropcols))
    non_numeric_columns = list(set(df.select_dtypes(['object']).columns).difference(dropcols))
    non_numeric_columns.append(None)
    options2 = st.selectbox('Что хотим сделать?',
             ['Посмотреть зависимости (EDA)', 'Кластеризация'], index=0)
    if options2 == 'Посмотреть зависимости (EDA)':
        visualisation(df, numeric_columns, non_numeric_columns)
    elif options2 == 'Кластеризация': 
        hierarch(df)
        
def hierarch(customers):
    df = pd.pivot_table(customers, index = ['CustomerID'], columns = ['Gender'], 
                    values = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
                    aggfunc = 'mean').fillna(0)
    df.columns = ['Female_age', 'Male_age', 'Female_income', 'Male_Income', 'Female_score', 'Male_score']
    X_embedded = TSNE(n_components=2, perplexity = 5, verbose = 0).fit_transform(df.to_numpy())
    vis_df = pd.DataFrame(X_embedded, index = df.index)
    fig = ff.create_dendrogram(np.array(vis_df[[0,1]]),
                               orientation='left',
                               linkagefun=lambda x: sch.linkage(np.array(vis_df[[0,1]]), "average"),)
    fig.update_layout(width=800, height=1600, font_size=8)
    fig.show()
        
        
options = st.selectbox('Какие данные скорить?',
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
    


    
# Title of the main page
st.title("Data Storyteller Application")

# Add all your applications (pages) here
#     df.sort_values(by=['id', 'time'], inplace=True)
#     res = scoring(df, '1406_best_model', new_cols2, tresh)
#     st.markdown('### Скоринг завершён успешно!')

#     st.markdown(get_table_download_link(res), unsafe_allow_html=True)
#     st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
#     slider_feats(res, 'id', target_col_name)

#         assert df.shape[1] == 3 or df.shape[1] == 4
#         st.markdown('#### Файл корректный!')  
#         st.write('Пример данных из файла:')
#         st.dataframe(df.sample(3))  
#         res = scoring(df, '1406_best_model', new_cols2, tresh)
#         st.markdown('### Скоринг завершён успешно!')

#         st.markdown(get_table_download_link(res), unsafe_allow_html=True)
#         st.write('Доля класса 1 = ', round(100*res[target_col_name].mean(), 1), ' %')
        
#         slider_feats(res, 'id', target_col_name)
        
        
# if st.sidebar.button('Анализ важности переменных модели'):
#     st.markdown('#### SHAP важности признаков модели')  
#     st.image("https://clip2net.com/clip/m392735/16318-clip-315kb.jpg?nocache=1")
    
# if st.sidebar.button('Анализ качества модели'):
#     st.markdown('#### Точность модели на train-val-test выборках:')  
#     st.image("https://clip2net.com/clip/m392735/c6560-clip-81kb.jpg?nocache=1")