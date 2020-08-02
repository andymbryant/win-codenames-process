import numpy as np
import pandas as pd
from scipy import spatial
from random import randint
from kneed import KneeLocator
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list, ward, optimal_leaf_ordering
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from statistics import mean
import json
import uuid

# CONSTANT = 6

def get_game_words(words_for_game_df):
    # Get sample from dataframe
    friend_series = words_for_game_df.sample(n=8)
    # Drop so there are no duplicates
    card_words = words_for_game_df.drop(friend_series.index)
    friends = [fr.lower() for fr in friend_series[0].tolist()]

    foe_series = words_for_game_df.sample(n=9)
    words_for_game_df = words_for_game_df.drop(foe_series.index)
    foes = [f.lower() for f in foe_series[0].tolist()]

    neutral_series = words_for_game_df.sample(n=7)
    words_for_game_df = words_for_game_df.drop(neutral_series.index)
    neutrals = [n.lower() for n in neutral_series[0].tolist()]

    assassin_series = words_for_game_df.sample(n=1)
    assassin = [a.lower() for a in assassin_series[0].tolist()]

    board_words = friends + foes + neutrals + assassin
    return friends, foes, neutrals, assassin, board_words

def distance(source, target, vectors):
    return spatial.distance.cosine(vectors.loc[source].to_numpy(), vectors.loc[target].to_numpy())

def get_top_friends(vectors, friends):
    data = np.array([vectors.loc[word].to_numpy() for word in friends])
    cos_data = pdist(data, metric='cosine')
    pairwise_cos = pd.DataFrame(
        squareform(cos_data),
        columns = friends,
        index = friends
    ).replace(to_replace=0, value=999)
    # How much should this be? More will favor larger clusters.
    MULT = 1.2
    avg_dist = mean(pairwise_cos.min()) * MULT
    Z = linkage(cos_data, method='complete', optimal_ordering=True)
    cos_indices = fcluster(Z, t=avg_dist, criterion='distance')
    np_friends = np.array(cos_indices)
    indices = np.where(np_friends == 1)[0]
    top_friends = [friends[i] for i in indices]

    low_friends = set(friends) - set(top_friends)
    return top_friends, low_friends

def get_scores(row, vectors, primary, **kwargs):
    word = row.word
    friends = kwargs.get('friends')
    board_words = kwargs.get('board_words')
    assassin = kwargs.get('assassin')
    top_friends = kwargs.get('top_friends')
    low_friends = kwargs.get('low_friends')
    foes = kwargs.get('foes')
    neutrals = kwargs.get('neutrals')

    # Initialize with na values
    goodness = assassin_dist = bad_minimax = neutrals_minimax = variance = np.nan
    na_series = pd.Series([goodness, bad_minimax, neutrals_minimax, variance])

    if word.lower() in board_words or any([bw in word.lower() for bw in board_words]):
        return na_series

    # If first candidates_df (glove) then check for assassin distance
    # If not, the assassin distance will be very different for the other models
    assassin_dist = [distance(word, a, vectors) for a in assassin]
    # Check if assassin distance is adequate, if not don't waste your time
    # Acceptable ~.8
    if primary and assassin_dist[0] <= 0.8:
        return na_series

    top_friends_dist = [distance(word, tf, vectors) for tf in top_friends]
    # changed this to to all friends, not low friends
    low_friends_dist = [distance(word, lf, vectors) for lf in low_friends]
    foes_dist = [distance(word, f, vectors) for f in foes]
    neutrals_dist = [distance(word, n, vectors) for n in neutrals]
    bad_dist = foes_dist + assassin_dist
    goodness = (sum(bad_dist)/len(bad_dist)) - (sum(top_friends_dist)/len(top_friends_dist))
    # OLD WAY OF CALCULATING GOODNESS
    # goodness = sum(foes_dist + assassin_dist) - CONSTANT * sum(top_friends_dist)
    friends_dist = top_friends_dist + low_friends_dist
    min_friends_dist = min(friends_dist)
    max_friends_dist = max(friends_dist)
    bad_minimax = min(bad_dist) - max_friends_dist
    neutrals_minimax = min(neutrals_dist) - max_friends_dist
    # Should this be absolute value?
    variance = abs(max_friends_dist) - abs(min_friends_dist)
    return pd.Series([goodness, bad_minimax, neutrals_minimax, variance])

def get_final_metrics(word, candidates, **kwargs):
    total_rank = 0
    total_variance = 0
    total_goodness = 0
    total_bad_minimax = 0
    total_neutrals_minimax = 0
    total_frequency = 0
    for candidate_df in candidates:
        word_select = candidate_df['word'] == word
        rank = candidate_df[word_select].index[0]
        total_rank = total_rank + rank
        variance = candidate_df[word_select].variance.iloc[0]
        total_variance = total_variance + variance
        goodness = candidate_df[word_select].goodness.iloc[0]
        total_goodness = total_goodness + goodness
        bad_minimax = candidate_df[word_select].bad_minimax.iloc[0]
        total_bad_minimax = total_bad_minimax + bad_minimax
        neutrals_minimax = candidate_df[word_select].neutrals_minimax.iloc[0]
        total_neutrals_minimax = total_neutrals_minimax + neutrals_minimax
        frequency = candidate_df[word_select].frequency.iloc[0]
        total_frequency = total_frequency + frequency
        # Multiply by 1.5 to favor the the high and penalize the low rank (lower number is higher rank)
    final_rank = total_rank * 1.5
    num_candidates = len(candidates)
    final_variance = total_variance / num_candidates
    final_goodness = total_goodness / num_candidates
    final_bad_minimax = total_bad_minimax / num_candidates
    final_neutrals_minimax = total_neutrals_minimax / num_candidates
    final_frequency = total_frequency / num_candidates
    return pd.Series([final_rank, final_goodness, final_bad_minimax, final_frequency, final_neutrals_minimax, final_variance])

def get_candidates_df(vectors, primary, **kwargs):
    words_to_consider = kwargs.get('words_to_consider')
    words_to_consider_frequencies = kwargs.get('words_to_consider_frequencies')
    candidates = pd.DataFrame({'word': words_to_consider, 'frequency': words_to_consider_frequencies})
    candidates[['goodness', 'bad_minimax', 'neutrals_minimax', 'variance']] = candidates.apply(lambda row: get_scores(row, vectors, primary, **kwargs), axis=1)
    candidates.dropna(inplace=True)
    sort_by_columns = ['goodness', 'bad_minimax', 'frequency', 'neutrals_minimax']
    candidates = candidates.sort_values(sort_by_columns, ascending=[False for i in range(len(sort_by_columns))]).reset_index(drop=True)
    return candidates

def create_records(final_candidates, **kwargs):
    top_friends = kwargs.get('top_friends')
    low_friends = kwargs.get('low_friends')
    foes = kwargs.get('foes')
    neutrals = kwargs.get('neutrals')
    assassin = kwargs.get('assassin')

    id_length = 8

    game_id = str(uuid.uuid4())[:id_length]
    clue_id = str(uuid.uuid4())[:id_length]

    # Generate clue record
    clue_record_output_path = f'./output/clues/clue_{clue_id}.json'
    clue_record = {
        'id': clue_id,
        'game_id': game_id,
        'clues': final_candidates.head(10).to_dict(orient='records')
    }
    with open(clue_record_output_path, 'w') as fp:
        json.dump(clue_record, fp)

    # Generate game record with formatted word dicts
    top_friends_formatted = [
        {
            'id': str(uuid.uuid4())[:id_length],
            'word': word,
            'type': 'friend',
            'is_top': True
        }
        for word in top_friends
    ]

    low_friends_formatted = [
        {
            'id': str(uuid.uuid4())[:id_length],
            'word': word,
            'type': 'friend',
            'is_top': False
        }
        for word in low_friends
    ]
    foes_formatted = [
        {
            'id': str(uuid.uuid4())[:id_length],
            'word': word,
            'type': 'foe',
            'is_top': None
        }
        for word in foes
    ]
    neutrals_formatted = [
        {
            'id': str(uuid.uuid4())[:id_length],
            'word': word,
            'type': 'neutral',
            'is_top': None
        }
        for word in neutrals
    ]
    assassin_formatted = [
        {
            'id': str(uuid.uuid4())[:id_length],
            'word': word,
            'type': 'assassin',
            'is_top': None
        }
        for word in assassin
    ]

    game_record_output_path = f'./output/game_record_{game_id}.json'
    game_record = {
        'id': game_id,
        'clue_id': clue_id,
        'words': top_friends_formatted + low_friends_formatted + foes_formatted + neutrals_formatted + assassin_formatted
    }
    with open(game_record_output_path, 'w') as fp:
        json.dump(game_record, fp)