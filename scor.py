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
import streamlit.components.v1 as componentss
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

    
def visualisation(df):
    chart_select = st.sidebar.selectbox(
        label="Выберите тип графиков",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )
    cat_features = ['gamecategory', 'subgamecategory', 'oblast', 'city',  'phone_ver',  'day', 'weekday']
    non_cat_features = ['os_android', 'timezone', 'hour', 'salary_rank',
 'salary_to_bucket', 'mean_salary', 'salary_change']
    numeric_columns = ['day', 'weekday', 'bundle_point', 'bundle_up', 'os_android', 'timezone', 'hour', 'salary_rank',
 'salary_to_bucket', 'mean_salary', 'salary_change']
    non_numeric_columns = ['Segment', 'gamecategory', 'subgamecategory', 'oblast', 'city',  'phone_ver', None]
    if chart_select == 'Scatterplots':
        st.sidebar.subheader("Scatterplot")
        x_values = st.sidebar.selectbox('Ось X', options=numeric_columns, index=0)
        y_values = st.sidebar.selectbox('Ось Y', options=numeric_columns, index=6)
        color_value = st.sidebar.selectbox("Цвет", options=non_numeric_columns)
        plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
        # display the chart
        st.plotly_chart(plot)
    if chart_select == 'Lineplots':
        st.sidebar.subheader("Линейные графики")
        x_values = st.sidebar.selectbox('Ось X', options=numeric_columns, index=6)
        y_values = st.sidebar.selectbox('Ось Y', options=numeric_columns, index=4)
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
    
@st.cache(suppress_st_warning=True)
# Подготовка данных для моделей
def preprocess(df_init):
  df = df_init.copy()
  df['bundle_point'] = df['bundle'].apply(lambda x: str(x).count('.'))
  df['bundle_up'] = df['bundle'].apply(lambda x: sum(map(str.isupper, str(x))))
  df['os'] = df['os'].apply(lambda x: str(x).lower())
  df['os_android'] = (df['os'] == 'android')
  df['timezone'] = df['shift'].apply(lambda x: str(x).split('+')[-1] if '+' in str(x) else 0)
  df['timezone'] = pd.to_numeric(df['timezone'])
  df['osv'] = df['osv'].apply(lambda x: str(x).replace('28', '9').replace('30', '11').replace('29', '10')\
                            .replace('(', '. ').replace(' ', '.').replace('27', '8').replace('26', '7')\
                            .replace('25', '6').replace('24', '5').replace('23', '4').replace('22', '3'))
  df['phone_ver'] = df.apply(lambda x: str(x['os']) + '_' + str(x['osv']).split('.')[0], axis=1)
  df['bundle_new'] = df['bundle'].apply(lambda x: str(x).replace('.com', '').replace('android','')\
                                        .replace('com.', '').replace('.', ' ').replace('org', '')
                                        )
  df['hour'] = pd.to_datetime(df['created']).dt.hour
  df['day'] = pd.to_datetime(df['created']).dt.day
  cities = pd.read_excel('data/cities.xlsx', sheet_name='city')
  df = df.merge(cities, on ='city', how='left')
  cities = pd.read_excel('data/cities.xlsx', sheet_name='oblast')
  df = df.merge(cities, on ='oblast', how='left')
  df['calday'] = pd.to_datetime(df.created).dt.date
  df.drop(['bundle', 'osv', 'shift', 'created', 'os'], axis=1, inplace=True)
  df['weekday'] = pd.to_datetime(df['calday']).dt.dayofweek
  for i in ['gamecategory', 'subgamecategory', 'oblast', 'city',
              'phone_ver', 'hour']:
    df[i] = df[i].fillna('NONE')
  df['hour_real'] = df['hour'] - df['timezone'] + 3
  return df


@st.cache(suppress_st_warning=True)
# Функция для tf-idf
def tfidf_feats(prep_df, is_train = True, postfix = 'v3'):
  if is_train:
    tfidf = TfidfVectorizer(analyzer='word', max_features = 50, norm = 'l2', ngram_range=(1,3))
    new_df = tfidf.fit_transform(prep_df['bundle_new'], prep_df['Segment'])
    joblib.dump(tfidf, f'models/tfidf_{postfix}')
  else:
    tfidf = joblib.load(f'models/tfidf_{postfix}')
    new_df = tfidf.transform(prep_df['bundle_new'])
  prep_df = prep_df.drop(['bundle_new'], axis=1)\
              .merge(pd.DataFrame(new_df.toarray(), columns= tfidf.get_feature_names_out()),
                left_index=True, right_index=True)
  return prep_df


@st.cache(suppress_st_warning=True)
def get_shap(loaded_model_cb, df):
    explainer = shap.TreeExplainer(loaded_model_cb)
    shap_values = explainer.shap_values(df)
    return explainer, shap_values

def interpret(df, model_number):
    from config import TARGET_COL, postfix
    loaded_model_cb = CatBoostClassifier()
    loaded_model_cb.load_model(f'models/model_{model_number}_{postfix}') 
    MODEL_COLS =  loaded_model_cb.feature_names_
 #   MODEL_COLS = joblib.load(f'models/cols_{model_number}') 
    df = tfidf_feats(df, is_train = False, postfix = postfix)
#     gbm = joblib.load('models/lightgbm')
#     lgb_train = lgb.Dataset(df[MODEL_COLS], df[TARGET_COL])
    explainer, shap_values = get_shap(loaded_model_cb, df[MODEL_COLS])

    pl.title('Assessing feature importance based on Shap values')
    shap.summary_plot(shap_values, df[MODEL_COLS],plot_type="bar",show=False)
    st.pyplot(bbox_inches='tight')
    pl.clf()
    ntree=st.number_input('Select the desired CustomerID for detailed explanation on the training set'
                                      , min_value=0
                                      , max_value=len(df)
                                      )
#     ntree=st.selectbox('Select the desired CustomerID for detailed explanation on the training set'
#                                        , options=ntree
#                                        )
#     graph = lgb.create_tree_digraph(gbm, tree_index=ntree, name='Tree54')
#     st.graphviz_chart(graph)



    predicted_values = loaded_model_cb.predict(df[MODEL_COLS])
    real_value = df[TARGET_COL]    
    shap.force_plot(explainer.expected_value, shap_values[ntree], # Убрать нолики
                    df[MODEL_COLS].iloc[ntree,:],matplotlib=True,show=False
                    ,figsize=(16,5))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()
    if st.button('Click here to see a drilldown of the SHAP values'):
        shap_table=pd.DataFrame(shap_values,columns=MODEL_COLS)
        st.table(shap_table.iloc[ntree])

def eda_func(df):
    if st.checkbox("Показать 10 строк датафрейма"):
        st.write(df.head(10))           
    col1, col2 = st.sidebar.columns(2)
    col1.image("https://i.ibb.co/Vwhhs7J/image.png", width=150)
    col2.image("https://c.radikal.ru/c03/2111/60/dc02f41efa07.png", width=150)

    if st.sidebar.button('Качество моделей'):
        st.image("data/roc.png")
    st.sidebar.subheader("Настройки визуализации")
    dropcols = ['CustomerID']
#     numeric_columns = list(set(df.select_dtypes(['float', 'int64']).columns).difference(dropcols))
#     non_numeric_columns = list(set(df.select_dtypes(['object']).columns).difference(dropcols))
#     non_numeric_columns.append(None)
    options2 = st.selectbox('Что хотим сделать?',
             ['Посмотреть зависимости (EDA)', 'Кластеризация', 'Интерпретация модели'], index=0)
    if options2 == 'Посмотреть зависимости (EDA)':
        visualisation(df)
#     elif options2 == 'Кластеризация': 
#         hierarch(df)
    elif options2 == 'Интерпретация модели': 
        model_number = st.selectbox('Выберите модель',
             ['Сегм 1 (25-31 Ж)', 'Сегм 2 (25-42 М - Пиво)', 'Сегм 3 (25-43 Ж - Дети)', 
              'Сегм 4 (18-44 МЖ - Животные)', 'Сегм 5 (18-45 МЖ)'], index=0)
        if model_number == 'Сегмент 1 (25-31 Ж)':
            interpret(df, 1)
        elif model_number == 'Сегмент 2 (25-42 М - Пиво)':
            interpret(df, 2)
        elif model_number == 'Сегмент 3 (25-43 Ж - Дети)':
            interpret(df, 3)
        elif model_number == 'Сегмент 4 (18-44 М и Ж - Животные)':
            interpret(df, 4)
        elif model_number == 'Сегмент 5 (18-45 М и Ж)':
            interpret(df, 5)       
        
        
        
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
col1, col2 = st.columns([3,1])
col1.image("https://i.ibb.co/Vwhhs7J/image.png", width=150)
col2.image("https://c.radikal.ru/c03/2111/60/dc02f41efa07.png", width=150)

# Title of the main page
st.title("Профилирование клиентов")   
     
options = st.selectbox('Какие данные анализируем?',
         ['Семпл данных', 'Загрузить новый датасет'], index=0)

# Считывание датафрейма
if options == 'Семпл данных':
    df = pd.read_csv('data/data_sample.csv')
    st.write('Data processing...')
    df = preprocess(df)
    eda_func(df)
else:
    file_buffer = st.file_uploader(label = 'Выберите датасет')
    if file_buffer:
        try:
            df = pd.read_csv(file_buffer, encoding=None)
            st.write('Data processing...')
            df = preprocess(df)
            eda_func(df)
        except:
            st.write('Файл некорректен!')
    

