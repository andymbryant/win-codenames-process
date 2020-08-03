import numpy as np
import pandas as pd
from scipy import spatial
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from statistics import mean
import json
import uuid
from datetime import datetime
from config import *

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
    avg_dist = mean(pairwise_cos.min()) * AVG_DIST_MULT
    Z = linkage(cos_data, method='complete', optimal_ordering=True)
    cos_indices = fcluster(Z, t=avg_dist, criterion='distance')
    np_friends = np.array(cos_indices)
    indices = np.where(np_friends == 1)[0]
    top_friends = [friends[i] for i in indices]

    low_friends = set(friends) - set(top_friends)
    return top_friends, low_friends

def get_words_dict(glove_vectors):
    # Get all words from glove vectors
    all_words = list(glove_vectors.index.get_level_values(level=0))
    words_for_game_df = pd.read_json(f'{VECTORS_OUTPUT_PATH}/words_v2.json')
    friends, foes, neutrals, assassin, board_words = get_game_words(words_for_game_df)
    top_friends, low_friends = get_top_friends(glove_vectors, friends)

    words_dict = {
        'friends': friends,
        'foes': foes,
        'neutrals': neutrals,
        'assassin': assassin,
        'board_words': board_words,
        'top_friends': top_friends,
        'low_friends': low_friends,
        'board_words': board_words,
        'words_to_consider': all_words,
        'words_to_consider_frequencies': [i for i in range(1, len(all_words) + 1)]
    }
    return words_dict

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
    # If too close to assasin, return na series
    assassin_dist = [distance(word, a, vectors) for a in assassin]
    if primary and assassin_dist[0] <= ASSASSIN_CUTOFF:
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
    final_rank = total_rank * RANK_MULT
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
    candidates = candidates.sort_values(PRIMARY_SORT_BY_COLUMNS, ascending=PRIMARY_SORT_ASCENDING).reset_index(drop=True)
    return candidates

def get_final_candidates_df(all_candidates, top_candidate_words):
    final_candidates = pd.DataFrame({'word': top_candidate_words})
    final_candidates[['rank', 'goodness', 'bad_minimax', 'frequency', 'neutrals_minimax', 'variance']] = final_candidates.word.apply(lambda word: get_final_metrics(word, all_candidates))
    # final_candidates = final_candidates.drop_duplicates(subset=['word'])
    # TODO Add svr prediction
    final_candidates = final_candidates.sort_values(SECONDARY_SORT_BY_COLUMNS, ascending=SECONDARY_SORT_ASCENDING).reset_index(drop=True)
    # Only return the top 5
    final_candidates = final_candidates.iloc[:5]
    return final_candidates

def create_game_record(final_candidates, **kwargs):
    top_friends = kwargs.get('top_friends')
    low_friends = kwargs.get('low_friends')
    foes = kwargs.get('foes')
    neutrals = kwargs.get('neutrals')
    assassin = kwargs.get('assassin')
    game_id = str(uuid.uuid4())[:ID_LENGTH]

    # Generate game record with formatted word dicts
    top_friends_formatted = [
        {
            'id': str(uuid.uuid4())[:ID_LENGTH],
            'word': word,
            'type': 'friend',
            'is_top_friend': True
        }
        for word in top_friends
    ]

    low_friends_formatted = [
        {
            'id': str(uuid.uuid4())[:ID_LENGTH],
            'word': word,
            'type': 'friend',
            'is_top_friend': False
        }
        for word in low_friends
    ]
    foes_formatted = [
        {
            'id': str(uuid.uuid4())[:ID_LENGTH],
            'word': word,
            'type': 'foe',
            'is_top_friend': None
        }
        for word in foes
    ]
    neutrals_formatted = [
        {
            'id': str(uuid.uuid4())[:ID_LENGTH],
            'word': word,
            'type': 'neutral',
            'is_top_friend': None
        }
        for word in neutrals
    ]
    assassin_formatted = [
        {
            'id': str(uuid.uuid4())[:ID_LENGTH],
            'word': word,
            'type': 'assassin',
            'is_top_friend': None
        }
        for word in assassin
    ]

    game_record_output_path = f'./output/games/game_{game_id}.json'
    game_record = {
        'id': game_id,
        'clues': final_candidates.head(10).to_dict(orient='records'),
        'process_config': {
            'AVG_DIST_MULT': AVG_DIST_MULT,
            'RANK_MULT': RANK_MULT,
            'PRIMARY_SORT_BY_COLUMNS': PRIMARY_SORT_BY_COLUMNS,
            'PRIMARY_SORT_ASCENDING': PRIMARY_SORT_ASCENDING,
            'SECONDARY_SORT_BY_COLUMNS': SECONDARY_SORT_BY_COLUMNS,
            'SECONDARY_SORT_ASCENDING': SECONDARY_SORT_ASCENDING,
            'HIGH_SIZE': HIGH_SIZE,
            'LOW_SIZE': LOW_SIZE,
            'ASSASSIN_CUTOFF': ASSASSIN_CUTOFF
        },
        'words': top_friends_formatted + low_friends_formatted + foes_formatted + neutrals_formatted + assassin_formatted
    }
    with open(game_record_output_path, 'w') as fp:
        json.dump(game_record, fp)

def get_new_row(final_candidates, **kwargs):
    new_row = [
        " ".join(list(final_candidates.word)[:10]),
        " ".join(kwargs.get('top_friends')),
        " ".join(kwargs.get('low_friends')),
        " ".join(kwargs.get('foes')),
        " ".join(kwargs.get('neutrals')),
        " ".join(kwargs.get('assassin'))
    ]
    return new_row

def create_output(output_df):
    now = datetime.now()
    time_info = now.strftime("%m-%d-%H-%M")
    output_path = f'./output/results/results_{time_info}.csv'
    output_df.to_csv(output_path)