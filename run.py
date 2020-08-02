import pandas as pd
from datetime import datetime
from config import HIGH_SIZE, LOW_SIZE
from helpers import *

def main():
    deep = True
    size = HIGH_SIZE if deep else LOW_SIZE
    num_games = 3

    # Import Glove vectors
    glove_vectors = pd.read_pickle('./processing/data/glove_vectors.pkl')

    # Initialize output dataframe for storing results
    output_df = pd.DataFrame(columns=['top_clues', 'top_friends', 'low_friends', 'foes', 'neutrals', 'assassin'])
    # Get all words from glove vectors
    all_words = list(glove_vectors.index.get_level_values(level=0))

    for i in range(num_games):
        print(f'Iteration: {i + 1}')
        # words_dict = get_words_dict()
        print('Start glove candidates')
        glove_candidates = get_candidates_df(glove_vectors, True, **words_dict)
        top_candidate_words = glove_candidates.word.tolist()[:size]
        all_candidates = [glove_candidates]
        if deep:
            google_vectors = pd.read_pickle('./processing/data/google_vectors.pkl')
            ft_vectors = pd.read_pickle('./processing/data/fasttext_vectors.pkl')
            words_dict['words_to_consider'] = top_candidate_words
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
            all_candidates.extend([google_candidates, ft_candidates])

        final_candidates = get_final_candidates_df(all_candidates, top_candidate_words)
        create_records(final_candidates, **words_dict)
        # Add row to csv output (for posterity)
        new_row = get_new_row(final_candidates, **words_dict)
        output.loc[i] = new_row

    create_output(output_df)

if __name__ == '__main__':
    main()
