from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import scipy
from sklearn.pipeline import Pipeline
from datetime import datetime
from bson import json_util
from joblib import dump
import pickle
import json
import numpy as np
import pymongo
import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

username = os.getenv('MONGO_USERNAME')
password = os.getenv('MONGO_PASSWORD')
MONGO_URI = f'mongodb://{username}:{password}@ds137267.mlab.com:37267/win_codenames?retryWrites=false'
client = MongoClient(MONGO_URI)
db = client.win_codenames
print('Connected to database...')

y_column = 'human_selected'
X_columns = ['rank', 'goodness', 'bad_minimax', 'frequency', 'neutrals_minimax', 'variance']
y_var = 'human_selected'
y_column = [y_var]
all_columns = X_columns + y_column
output = pd.DataFrame(columns=all_columns)
all_reviews = [review for review in db.reviews.find()]

now = datetime.now()
time_info = now.strftime("%m-%d-%H-%M")

# Uncomment to save reviews locally
# json_filepath = f'./output/reviews_pkl/reviews_{time_info}.pkl'
# with open(json_filepath, 'wb') as outfile:
#     pickle.dump(all_reviews, outfile)

print(f'{len(all_reviews)} reviews found...')

for review in all_reviews:
    game_id = review['game_id']
    game = dict(db.games.find_one({'id': game_id}))

    for clue in game['clues']:
        new_row = clue
        new_row[y_var] = 1 if review[y_var] == clue['word'] else 0
        output.loc[len(output)] = new_row

print('Preparing data...')
X = output[X_columns].values.tolist()
y = output[y_var].values.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# I tried a few different methods, all with the hope of informing the final sort
# For each I set up a pipeline and used a grid search for the best parameters

# 1. Scaler, PCA, and SVM

# steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SVM', SVC())]
# pipe = Pipeline(steps)
# parameters = {
#     'SVM__C':[0.001,0.1,10,20, 50],
#     'SVM__gamma':[100,75,50,35,20,10,5,2,1],
#     'pca__n_components': [1,2,3,4,5,6]
#     }
# grid = GridSearchCV(pipe, param_grid=parameters, cv=10)
# print('Starting grid fit...')
# grid.fit(X_train, y_train)
# grid_score = grid.score(X_test,y_test)
# grid_best_params = grid.best_params_
# print(f'Grid score: {grid_score}')
# print(f'Grid params: {grid_best_params}')
# Grid score: 0.8159509202453987
# Grid params: {'SVM__C': 10, 'SVM__gamma': 5}

# steps = [('scaler', StandardScaler()), ('pca', PCA(n_components=3)), ('SVM', SVC(kernel='rbf', C=10, gamma=5))]
# pipe = Pipeline(steps)
# print('Training model...')
# pipe.fit(X_train, y_train)
# y_pred = pipe.predict(X_test)
# print(f'y_test: {y_test}')
# print(f'y_pred: {y_pred}')
# f1 = f1_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'F1: {f1}, Accuracy: {accuracy}')
# F1 ~0.24, Accuracy ~0.8

# 2. PCA combined with logistic regression

# pipe = Pipeline([('pca', PCA(n_components=3)), ('logistic', LogisticRegression())])
# pipe.fit(X_train, y_train)
# y_pred = pipe.predict(X_test)
# f1 = f1_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'F1: {f1}, Accuracy: {accuracy}')
# F1 0, Accuracy 0.3

# 3. SVR
# Given that my f1 score was so low, I thought it'd be better to include a probability in wholistic sorting algorithm
# rather than rely on strict classification

# steps = [('SVR', SVR())]
# pipe = Pipeline(steps)
model = SVR()
# parameters = {
#     'SVR__C':[0.0001,0.001,0.1,1,2,5,10,20,35,50],
#     'SVR__epsilon':[20,10,5,1,0.5,0.2,0.1,0.01,0.001],
#     'SVR__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'SVR__gamma': [100,75,50,35,20,10,5,2,1]
#     }
parameters = {
    'C': scipy.stats.expon(scale=100),
    # 'epsilon': [20,10,5,1,0.5,0.2,0.1,0.01,0.001],
    'epsilon': scipy.stats.expon(scale=.1),
    'gamma': scipy.stats.expon(scale=.1),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
# grid = GridSearchCV(pipe, param_grid=parameters, cv=10)
n_iter = 5
cv = 1
search = RandomizedSearchCV(model, param_distributions=parameters, cv=cv, n_iter=n_iter)

search.fit(X_train, y_train)
print(f'Best params: {search.best_params_}')

# model = SVR(C=0.1, epsilon=0.2)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# print(f'R2: {r2}, MSE: {mse}')

# model_output_path = f'./output/models/svr_{time_info}.joblib'
# dump(model, model_output_path)


