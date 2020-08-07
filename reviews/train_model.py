from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, SCORERS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from scipy import stats
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
from glob import glob

load_dotenv()

DOWNLOAD_REVIEWS = True
DOWNLOAD_GAMES = True
SAVE_GAMES = True
SAVE_REVIEWS = True
TUNE_PARAMS = False
SAVE_MODEL = True

now = datetime.now()
time_info = now.strftime("%m-%d-%H-%M")

y_column = 'human_selected'
X_columns = ['rank', 'goodness', 'bad_minimax', 'frequency', 'neutrals_minimax', 'variance']
y_var = 'human_selected'
y_column = [y_var]
all_columns = X_columns + y_column
output = pd.DataFrame(columns=all_columns)

if DOWNLOAD_REVIEWS or DOWNLOAD_GAMES:
    username = os.getenv('MONGO_USERNAME')
    password = os.getenv('MONGO_PASSWORD')
    MONGO_URI = f'mongodb://{username}:{password}@ds137267.mlab.com:37267/win_codenames?retryWrites=false'
    client = MongoClient(MONGO_URI)
    db = client.win_codenames

if DOWNLOAD_REVIEWS:
    print('Connected to database...')
    all_reviews = [review for review in db.reviews.find()]
    if SAVE_REVIEWS:
        json_filepath = f'./output/reviews_pkl/reviews_{time_info}.pkl'
        with open(json_filepath, 'wb') as outfile:
            pickle.dump(all_reviews, outfile)
else:
    list_of_files = glob('./output/reviews_pkl/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'rb') as pickle_file:
        all_reviews = pickle.load(pickle_file)

if DOWNLOAD_GAMES:
    all_games = []
    for review in all_reviews:
        game_id = review['game_id']
        game = dict(db.games.find_one({'id': game_id}))
        all_games.append(game)
    if SAVE_GAMES:
        json_filepath = f'./output/games_pkl/games_{time_info}.pkl'
        with open(json_filepath, 'wb') as outfile:
            pickle.dump(all_games, outfile)
else:
    list_of_files = glob('./output/games_pkl/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'rb') as pickle_file:
        all_games = pickle.load(pickle_file)
print(f'{len(all_reviews)} reviews found...')
for review in all_reviews:
    game_id = review['game_id']
    game = [game for game in all_games if game['id'] == game_id][0]

    for clue in game['clues']:
        new_row = clue
        new_row[y_var] = 1 if review[y_var] == clue['word'] else 0
        output.loc[len(output)] = new_row

print('Preparing data...')
X = output[X_columns].values.tolist()
y = output[y_var].values.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Very helpful read
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
if TUNE_PARAMS:
    grid_params = {
        'C': np.linspace(0.0001,2,21),
        'epsilon': np.linspace(0,1,20),
        'gamma': np.linspace(0,2,20)
    }

    cv = 5
    model = SVR()
    print('Start search...')
    search = GridSearchCV(model, param_grid=grid_params, cv=cv)
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print(f'Best params: {search.best_params_}')
    C = best_params['C']
    epsilon = best_params['epsilon']
    gamma = best_params['gamma']
    # Multiple iterations
    # Best params: {'C': 0.32435022868815844, 'epsilon': 0.1, 'gamma': 2.0, 'kernel': 'rbf'}
    # Best params: {'C': 0.32435022868815844, 'epsilon': 0.08163265306122448, 'gamma': 2.0, 'kernel': 'rbf'}
    # Best params: {'C': 0.2378613016572934, 'epsilon': 0.10526315789473684, 'gamma': 'auto'}
    # Best params: {'C': 0.300085, 'epsilon': 0.10526315789473684, 'gamma': 1.0}
    # Best params {'C': 0.300085, 'epsilon': 0.10526315789473684, 'gamma': 0.8421052631578947}
else:
    C = 0.3
    epsilon = 0.1
    gamma = 0.00001
    # C = 0.1
    # epsilon = 0.2
    # gamma = 'scale'

model = SVR(C=C, epsilon=epsilon, gamma=gamma)
print('Fitting model...')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'R2: {r2}, MSE: {mse}')

if SAVE_MODEL:
    print('Saving model...')
    model_output_path = f'./output/models/svr_{time_info}.joblib'
    dump(model, model_output_path)


