from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from datetime import datetime
from bson import json_util
import pickle
import json
import numpy as np
import pymongo
import os
import pandas as pd
import pickle
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
json_filepath = f'./output/reviews_pkl/reviews_{time_info}.pkl'
with open(json_filepath, 'wb') as outfile:
    pickle.dump(all_reviews, outfile)

print(f'{len(all_reviews)} reviews found...')

for review in all_reviews:
    game_id = review['game_id']
    game = dict(db.games.find_one({'id': game_id}))

    for clue in game['clues']:
        new_row = clue
        new_row[y_var] = 1 if review[y_var] == clue['word'] else 0
        output.loc[len(output)] = new_row

X_columns = all_columns
X_columns.remove(y_var)

print('Preparing data...')
X = output[X_columns].values.tolist()
y = output[y_var].values.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
print('Training model...')
model.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()), ('svr', SVR(epsilon=0.2))])

y_pred = model.predict(X_test)
print(f'Predictions: {y_pred}')
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}, R2: {r2}')
