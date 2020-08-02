import pandas as pd
from config import HIGH_SIZE, LOW_SIZE
from helpers import *

def main():
    # If True, GloVe Google and FT vectors will be used with a larger group of words
    # If False, only GloVe vectors will be used on a smaller subset of words
    deep = True
    # If True a result csv will be created
    create_result_csv = True
    size = HIGH_SIZE if deep else LOW_SIZE
    # Number of games to be generated
    num_games = 3

    # Import Glove vectors
    glove_vectors = pd.read_pickle('./processing/data/glove_vectors.pkl')

    if create_result_csv:
        output_df = pd.DataFrame(columns=['top_clues', 'top_friends', 'low_friends', 'foes', 'neutrals', 'assassin'])

    for i in range(num_games):
        print(f'Generating game: {i + 1}')
        words_dict = get_words_dict(glove_vectors)
        print('Starting GloVe candidates...')
        glove_candidates = get_candidates_df(glove_vectors, True, **words_dict)
        top_candidate_words = glove_candidates.word.tolist()[:size]
        all_candidates = [glove_candidates]
        if deep:
            google_vectors = pd.read_pickle('./processing/data/google_vectors.pkl')
            ft_vectors = pd.read_pickle('./processing/data/fasttext_vectors.pkl')
            words_dict['words_to_consider'] = top_candidate_words
            words_dict['words_to_consider_frequencies'] = [glove_candidates[glove_candidates.word == word].frequency.iloc[0] for word in top_candidate_words]
            print('Starting Google candidates...')
            google_candidates = get_candidates_df(google_vectors, False, **words_dict)
            print('Starting FT candidates...')
            ft_candidates = get_candidates_df(ft_vectors, False, **words_dict)
            top_glove_words = set(glove_candidates.word.tolist())
            top_google_words = set(google_candidates.word.tolist())
            top_ft_words = set(ft_candidates.word.tolist())
            # Only consider words that are in all three outputs
            top_candidate_words = list(top_glove_words.intersection(top_google_words, top_ft_words))
            all_candidates.extend([google_candidates, ft_candidates])

        print('Merging all candidates...')
        final_candidates = get_final_candidates_df(all_candidates, top_candidate_words)
        create_records(final_candidates, **words_dict)
        # Add row to csv output (for posterity)
        if create_result_csv:
            new_row = get_new_row(final_candidates, **words_dict)
            output_df.loc[i] = new_row

    if create_result_csv:
            # Initialize output dataframe for storing results
        create_output(output_df)

if __name__ == '__main__':
    main()
