import numpy as np
import pandas as pd
from scipy import spatial
from random import randint
from kneed import KneeLocator
# import nbconvert.filters.strings
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

CONSTANT = 4

def get_game_words(words_for_game_df):
    # Get sample from dataframe
    friend_series = words_for_game_df.sample(n=8)
    # Drop so there are no duplicates
    card_words = words_for_game_df.drop(friend_series.index)
    friends = [fr.lower() for fr in friend_series[0].tolist()]

    foe_series = words_for_game_df.sample(n=9)
    words_for_game_df = words_for_game_df.drop(foe_series.index)
    foes = [f.lower() for f in foe_series[0].tolist()]

    neutral_series = words_for_game_df.sample(n=6)
    words_for_game_df = words_for_game_df.drop(neutral_series.index)
    neutrals = [n.lower() for n in neutral_series[0].tolist()]

    assassin_series = words_for_game_df.sample(n=1)
    assassin = [a.lower() for a in assassin_series[0].tolist()]

    board_words = friends + foes + neutrals + assassin
    return friends, foes, neutrals, assassin, board_words

def distance(source, target, vectors):
    return spatial.distance.cosine(vectors.loc[source].to_numpy(), vectors.loc[target].to_numpy())

def get_top_friends(vectors, friends):
    # Get optimal friends
    data = np.array([vectors.loc[word].to_numpy() for word in friends])
    pairwise_cos = pd.DataFrame(
        squareform(pdist(data, metric='cosine')),
        columns = friends,
        index = friends
    ).replace(to_replace=0, value=999)
    data_dict = dict(pairwise_cos.min())
    y = list(data_dict.values())
    x = range(1, len(data_dict.keys())+1)
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    if kn.knee > 1:
        num_friends_to_consider = kn.knee
    else:
        # Randomly make it 4 sometimes?
        num_friends_to_consider = 2
    top_friends = sorted(data_dict, key=data_dict.get, reverse=False)[:num_friends_to_consider]
    print(f'Friends: {friends}')
    print(f'Top friends: {top_friends}')
    low_friends = set(friends) - set(top_friends)
    return top_friends, low_friends

def get_scores(row, vectors, **kwargs):
    word = row.word
    friends = kwargs.get('friends')
    board_words = kwargs.get('board_words')
    assassin = kwargs.get('assassin')
    top_friends = kwargs.get('top_friends')
    low_friends = kwargs.get('low_friends')
    foes = kwargs.get('foes')
    neutrals = kwargs.get('neutrals')
    if word in board_words or any([bw in word for bw in board_words]):
        goodness = assassin_minimax = foes_minimax = neutrals_minimax = variance = -1000
    else:
        assassin_dist = [distance(word, a, vectors) for a in assassin]
        # Check if assassin distance is adequate, if not don't waste your time
        if abs(assassin_dist[0]) > 0.001:
            top_friends_dist = [distance(word, tf, vectors) for tf in top_friends]
            all_friends_dist = [distance(word, lf, vectors) for lf in low_friends]
            foes_dist = [distance(word, f, vectors) for f in foes]
            neutrals_dist = [distance(word, n, vectors) for n in neutrals]
            goodness = sum(foes_dist + assassin_dist) - CONSTANT * sum(top_friends_dist)
            min_friends_dist = min(all_friends_dist)
            max_friends_dist = max(all_friends_dist)
            assassin_minimax = min(assassin_dist) - max_friends_dist
            foes_minimax = min(foes_dist) - max_friends_dist
            neutrals_minimax = min(neutrals_dist) - max_friends_dist
            # Should this be absolute value?
            variance = abs(max_friends_dist) - abs(min_friends_dist)
    return pd.Series([goodness, assassin_minimax, foes_minimax, neutrals_minimax, variance])

def get_final_metrics(word, candidates, **kwargs):
    total_rank = 0
    total_variance = 0
    total_goodness = 0
    for candidate_df in candidates:
        word_select = candidate_df['word'] == word
        rank = candidate_df.index[word_select].tolist()[0]
        total_rank = total_rank + rank
        variance = candidate_df[word_select].variance.tolist()[0]
        total_variance = total_variance + variance
        goodness = candidate_df[word_select].goodness.tolist()[0]
        total_goodness = total_goodness + goodness
    return pd.Series([total_rank * 1.5, (total_variance/len(candidates)), (total_goodness/len(candidates))])

def get_candidates_df(vectors, **kwargs):
    all_words = kwargs.get('all_words')
    candidates = pd.DataFrame({'word': all_words, 'frequency': [i for i in range(1, len(all_words) + 1)]})
    candidates[['goodness', 'assassin_minimax', 'foes_minimax', 'neutrals_minimax', 'variance']] = candidates.apply(lambda row: get_scores(row, vectors, **kwargs), axis=1)
    sort_by_columns = ['goodness', 'foes_minimax', 'assassin_minimax', 'frequency', 'neutrals_minimax', 'variance']
    return candidates.sort_values(sort_by_columns, ascending=[False for i in range(len(sort_by_columns))]).reset_index(drop=True)