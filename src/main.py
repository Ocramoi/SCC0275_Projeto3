#!/usr/bin/env python3

## Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.neural_network import MLPRegressor

from typing import Tuple, Dict, List

## Constants
DATADIR = "../data/"

# Q1

## Q1a
_dfDay = pd.read_csv(DATADIR + "day.csv")
_dfHour = pd.read_csv(DATADIR + "hour.csv")

print("Day:")
print(_dfDay.head())

print("Hour:")
print(_dfHour.head())

## Q1b
# A variável resposta é a da coluna `cnt`, que indica o número total de bikes
# alugadas (casual + registered)
_dfHour['cnt'].hist()
yColumns = [ 'cnt' ]
plt.show()

sns.pairplot(_dfHour[['temp', 'cnt']])
plt.show()

# Q2

## Q2a
#  Do total de colunas:
print(", ".join(_dfHour.columns))
# Temos como colunas explicativas
"""
- season
- holiday
- weekday
- workingday
- weathersit
- temp
- atemp
- hum
- windspeedt
"""
XColumns = ["season", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]

## Q2b

## Q2c
"""
- instant
- dteday
- yr
- mnth
- hr
"""

# Q3

## Q3a
Train = _dfHour[_dfHour['yr'] == 0]
XTrain, yTrain = Train[XColumns], Train[yColumns]
yTrain = np.array(yTrain).reshape((-1, ))

Test = _dfHour[_dfHour['yr'] == 1]
XTest, yTest = Test[XColumns], Test[yColumns]
yTest = np.array(yTest).reshape((-1, ))

## Q3b
QUALI = ['season', 'weathersit',]
transformer = make_column_transformer(
    (OneHotEncoder(), QUALI),
    remainder='passthrough',
    verbose_feature_names_out=False,
)

trainTransformed = transformer.fit_transform(XTrain)
XTrain = pd.DataFrame(trainTransformed, columns=transformer.get_feature_names_out())
testTransformed = transformer.fit_transform(XTest)
XTest = pd.DataFrame(testTransformed, columns=transformer.get_feature_names_out())

## Q4
import concurrent.futures
from sklearn.metrics import mean_absolute_error

MAX_ITERS = int(5e3)
hiddenSizes: List[Tuple[int, int]] = [(), (10,), (10, 10,)]
models: List[Tuple[Tuple[int, int], MLPRegressor]] = []

def createFitModel(size: Tuple[int, int]) -> Tuple[Tuple[int, int], MLPRegressor]:
    regr = MLPRegressor(hidden_layer_sizes=size, random_state=42, max_iter=MAX_ITERS)
    regr.fit(XTrain, yTrain)
    return (size, regr)

with concurrent.futures.ThreadPoolExecutor() as executor:
    models = list(executor.map(createFitModel, hiddenSizes))
    print("Modelos gerados")

for model in models:
    print(
        "Modelo com camadas {} - MAE: {}".format(
            model[0],
            mean_absolute_error(model[1].predict(XTest), yTest)
        )
    )
