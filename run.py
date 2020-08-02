import pandas as pd
from datetime import datetime
from helpers import *

def main():

    # Make express even more express
    # Increase threshold of assassin for passing
    # I should probably change it from assassin as the threshold - high friends?
    # Also no need to combine the DataFrame
    deep = True
    num_iterations = 1

    # Import Glove vectors
    glove_vectors = pd.read_pickle('./processing/data/glove_vectors.pkl')

    # Initialize output dataframe for storing results
    output = pd.DataFrame(columns=['top_clues', 'top_friends', 'low_friends', 'foes', 'neutrals', 'assassin'])
    # Get all words from glove vectors
    all_words = list(glove_vectors.index.get_level_values(level=0))

    for i in range(num_iterations):
        print(f'Iteration: {i + 1}')
        # Import words for game and make categorized lists from them (i.e. generate cards for game)
        words_for_game_df = pd.read_json('./processing/data/words_v2.json')
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

        size = 2000 if deep else 250
        print('Start glove candidates')
        glove_candidates = get_candidates_df(glove_vectors, True, **words_dict)
        top_candidate_words = glove_candidates.word.tolist()[:size]
        all_candidates = [glove_candidates]
        if deep:
            google_vectors = pd.read_pickle('./processing/data/google_vectors.pkl')
            ft_vectors = pd.read_pickle('./processing/data/fasttext_vectors.pkl')
            # print('Printing top glove candidates.')
            words_dict['words_to_consider'] = top_candidate_words
            # Get word frequencies from glove
            words_dict['words_to_consider_frequencies'] = [glove_candidates[glove_candidates.word == word].frequency.iloc[0] for word in top_candidate_words]
            print('Start google candidates')
            google_candidates = get_candidates_df(google_vectors, False, **words_dict)
            print('Start ft candidates')
            ft_candidates = get_candidates_df(ft_vectors, False, **words_dict)
            top_glove_words = set(glove_candidates.word.tolist())
            top_google_words = set(google_candidates.word.tolist())
            top_ft_words = set(ft_candidates.word.tolist())
            # Only consider words that are in all three outputs
            top_candidate_words = list(top_glove_words.intersection(top_google_words, top_ft_words))
            all_candidates = [glove_candidates, google_candidates, ft_candidates]

        final_candidates = pd.DataFrame({'word': top_candidate_words})
        print('starting get final metrics')
        final_candidates[['rank', 'variance', 'goodness']] = final_candidates.word.apply(lambda word: get_final_metrics(word, all_candidates))
        # final_candidates = final_candidates.drop_duplicates(subset=['word'])
        final_candidates = final_candidates.sort_values(['rank', 'variance', 'goodness'], ascending=[True,True,False]).reset_index(drop=True)
        print('final candidates:')
        print(final_candidates)
        new_row = [
            " ".join(list(final_candidates.word)[:10]),
            " ".join(top_friends),
            " ".join(low_friends),
            " ".join(foes),
            " ".join(neutrals),
            " ".join(assassin)
        ]
        output.loc[i] = new_row
    now = datetime.now()
    time_info = now.strftime("%m-%d-%H-%M")
    output_path = f'./output/output_{time_info}.csv'
    output.to_csv(output_path)

if __name__ == '__main__':
    main()
