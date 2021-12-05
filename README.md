***Команда Fit_Predict***

***@SlayZar, @colindonolwe, @mktoid***

# Определение профилей аудитории HyperAdTech

- Фичи по зарплате в городе

- TF-IDF фичи по текстовому индификатору приложения

- Модели градиентного бустинга CatBoost


## Классификация

### Модель на применимых переменных

Качество модели по метрике ROC-AUC на валидационном датасете:

| Сегмент 1   | Сегмент 2   | Сегмент 3   | Сегмент 4   | Сегмент 5  |
| ----------- |:-----------:| -----------:| -----------:| ----------:|
| 0.73        | 0.86        | 0.73        | 0.76        | 0.81       |


https://colab.research.google.com/drive/1rBO6iUHpPrhnM7p-FlTOPa6huwhYWd3-?usp=sharing

<img src="/data/roc.png" width="500" />

### Модель с лучшим качеством на предоставленной выборке

| Сегмент 1   | Сегмент 2   | Сегмент 3   | Сегмент 4   | Сегмент 5  |
| ----------- |:-----------:| -----------:| -----------:| ----------:|
| 0.9         | 0.95        | 0.82        | 0.84        | 0.91       |

https://colab.research.google.com/drive/1EU3gNqhcvugnI6j9hngQ3T8aTZW5owjN?usp=sharing


<img src="/data/good_roc.png" width="500" />


## Кластеризация


<img src="/data/cluster.png" width="500" />

https://colab.research.google.com/drive/1HskvXXE_d42xVDES_SsfHe1_kjmoVe1u?usp=sharing

## Презентация

https://docs.google.com/presentation/d/1KRlx6Urw1KaVmDWVdt-ahjXHzeYDYnrO3N6bTow_hlw/edit?usp=sharing

## Демо

https://share.streamlit.io/slayzar/final_rsv_2021_fit_predict/main/scor.py
