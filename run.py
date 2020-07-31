import pandas as pd
# import nbconvert.filters.strings
from helpers import *

def main():

    express = False
    size = 25
    num_iterations = 3

    # Import Glove vectors
    glove_vectors = pd.read_pickle('./processing/data/glove_vectors.pkl')

    # Initialize output dataframe for storing results
    output = pd.DataFrame(columns=['top_clues', 'top_friends', 'low_friends', 'foes', 'neutrals', 'assassin'])
    # Get all words from glove vectors
    all_words = list(glove_vectors.index.get_level_values(level=0))

    # Import words for game and make categorized lists from them
    words_for_game = pd.read_json('./processing/data/words_v2.json')
    friends, foes, neutrals, assassin, board_words = get_game_words(words_for_game)
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
        'all_words': all_words
    }

    for i in range(num_iterations):
        if express:
            glove_candidates = get_candidates_df(glove_vectors, **words_dict)
            top_candidate_words = glove_candidates.word.tolist()[:size]
            all_candidates = [glove_candidates]
        else:
            google_vectors = pd.read_pickle('./processing/data/google_vectors.pkl')
            ft_vectors = pd.read_pickle('./processing/data/fasttext_vectors.pkl')
            glove_candidates = get_candidates_df(glove_vectors, **words_dict)
            google_candidates = get_candidates_df(google_vectors, **words_dict)
            ft_candidates = get_candidates_df(ft_vectors, **words_dict)
            top_candidate_words = glove_candidates.word.tolist()[:size] + google_candidates.word.tolist()[:size] + ft_candidates.word.tolist()[:size]
            all_candidates = [glove_candidates, google_candidates, ft_candidates]

        final_candidates = pd.DataFrame({'word': top_candidate_words})
        final_candidates[['rank', 'variance', 'goodness']] = final_candidates.word.apply(lambda word: get_final_metrics(word, all_candidates))
        final_candidates = final_candidates.drop_duplicates(subset=['word'])
        final_candidates.sort_values(['rank', 'variance', 'goodness'], ascending=[True,True,False]).reset_index(drop=True)


        new_row = [
            " ".join(list(final_candidates.word)[:10]),
            " ".join(top_friends),
            " ".join(low_friends),
            " ".join(foes),
            " ".join(neutrals),
            " ".join(assassin)
        ]
        output.loc[i] = new_row
    output.to_csv('./output/output.csv')

if __name__ == '__main__':
    main()
