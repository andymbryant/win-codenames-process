import pandas as pd
from config import SIZE, VECTORS_OUTPUT_PATH, DEEP
from helpers import *

def main():
    # If True a result csv will be created
    create_result_csv = True
    # Number of games to be generated
    num_games = 100

    # Import Glove vectors
    glove_vectors = pd.read_pickle(f'{VECTORS_OUTPUT_PATH}/glove_vectors.pkl')

    if create_result_csv:
        output_df = pd.DataFrame(columns=['top_clues', 'top_friends', 'low_friends', 'foes', 'neutrals', 'assassin'])

    for i in range(num_games):
        print(f'Generating game: {i + 1}')
        words_dict = get_words_dict(glove_vectors)
        print(words_dict['top_friends'])
        print(words_dict['assassin'])
        # print(words_dict['friends'])
        # print(words_dict['foes'])
        print('Starting GloVe candidates...')
        glove_candidates = get_candidates_df(glove_vectors, True, **words_dict)
        top_candidate_words = glove_candidates.word.tolist()[:SIZE]
        all_candidates = [glove_candidates]
        if DEEP:
            google_vectors = pd.read_pickle(f'{VECTORS_OUTPUT_PATH}/google_vectors.pkl')
            ft_vectors = pd.read_pickle(f'{VECTORS_OUTPUT_PATH}/fasttext_vectors.pkl')
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
        create_game_record(final_candidates, **words_dict)
        # Add row to csv output (for posterity)
        if create_result_csv:
            new_row = get_new_row(final_candidates, **words_dict)
            output_df.loc[i] = new_row

    if create_result_csv:
        # Initialize output dataframe for storing results
        create_output(output_df)

if __name__ == '__main__':
    main()
