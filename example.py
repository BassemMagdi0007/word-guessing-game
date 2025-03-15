"""
    python example.py env/simple-env.json
"""
import pandas as pd
import string
import math
from collections import defaultdict
#------------------------------------------------------------------------------------------------------
"""Data preprocessing"""
# Load the pre-filtered city data
cities_df = pd.read_csv('filtered_worldcities.csv')

# Convert city names to uppercase for consistency
cities_df['city'] = cities_df['city'].str.upper()

# Separate Tanzanian and non-Tanzanian cities
tanzanian_cities = cities_df[cities_df['iso2'] == 'TZ']['city'].tolist()
non_tanzanian_cities = cities_df[cities_df['iso2'] != 'TZ']['city'].tolist()
all_cities = tanzanian_cities + non_tanzanian_cities

#------------------------------------------------------------------------------------------------------
"""Calculate the information gain from guessing a letter."""
def calculate_information_gain(possible_words, probabilities, letter):
    if len(possible_words) == 0:
        return 0

    # Current entropy (using actual probabilities)
    current_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    # Group words by their pattern and track their probabilities
    pattern_probs = defaultdict(float)
    pattern_words = defaultdict(list)
    for word, prob in zip(possible_words, probabilities):
        pattern = "".join([letter if char == letter else '-' for char in word])
        pattern_probs[pattern] += prob
        pattern_words[pattern].append((word, prob))

    # Calculate expected entropy after guessing the letter
    expected_entropy = 0
    for pattern, total_prob in pattern_probs.items():
        if total_prob == 0:
            continue
        # Calculate entropy for this pattern
        conditional_probs = [prob / total_prob for (_, prob) in pattern_words[pattern]]
        entropy = -sum(p * math.log2(p) for p in conditional_probs if p > 0)
        expected_entropy += total_prob * entropy

    return current_entropy - expected_entropy
#------------------------------------------------------------------------------------------------------
"""Filter the word list based on feedback and previous guesses."""
def filter_words(word_list, feedback, guesses):
    filtered_list = []
    letter_guesses = [g for g in guesses if len(g) == 1]
    
    for word in word_list:
        if len(word) != len(feedback):
            continue
            
        is_match = True
        
        # Check if revealed positions match
        for i, (char, feedback_char) in enumerate(zip(word, feedback)):
            if feedback_char != '-' and char != feedback_char:
                is_match = False
                break
        
        if not is_match:
            continue
            
        # Check if the word contains guessed letters that aren't in the feedback
        for letter in letter_guesses:
            if letter in word and letter not in feedback:
                is_match = False
                break
                
        # Check if unrevealed positions have letters that should be revealed
        for i, char in enumerate(word):
            if feedback[i] == '-' and char in feedback:
                # If this letter appears in the feedback but not at this position,
                # this word doesn't match the pattern
                is_match = False
                break
                
        if is_match:
            filtered_list.append(word)
            
    return filtered_list

#------------------------------------------------------------------------------------------------------
"""Update word probabilities based on the Tanzanian bias."""
def update_word_probabilities(word_list):
    if not word_list:
        return []
    probabilities = []
    tz_cities = [city for city in word_list if city in tanzanian_cities]
    non_tz_cities = [city for city in word_list if city in non_tanzanian_cities]
    
    for word in word_list:
        if word in tanzanian_cities:
            prob = 0.5 / len(tz_cities) if len(tz_cities) > 0 else 0
        else:
            prob = 0.5 / len(non_tz_cities) if len(non_tz_cities) > 0 else 0
        probabilities.append(prob)
    
    return probabilities

#------------------------------------------------------------------------------------------------------
# Cache for initial guesses based on word length
initial_guess_cache = {}

"""Get the best initial guess for a given word length."""
def get_initial_guess(word_length, feedback="", guesses=[]):
    # Filter cities by length and feedback
    possible_words = [city for city in all_cities if len(city) == word_length]
    possible_words = filter_words(possible_words, feedback, guesses)

    # Calculate probabilities
    probabilities = update_word_probabilities(possible_words)

    # Find the letter with the highest information gain
    best_letter = None
    max_gain = -1
    for letter in string.ascii_uppercase:
        if letter not in guesses:  # Avoid repeating guesses
            gain = calculate_information_gain(possible_words, probabilities, letter)
            if gain > max_gain:
                max_gain = gain
                best_letter = letter

    return best_letter

#------------------------------------------------------------------------------------------------------
def select_best_letter(possible_words, probabilities, feedback, guesses):
    candidate_letters = set(letter for word in possible_words for letter in word) - set(guesses)
    
    if not candidate_letters:
        # Fallback to unguessed letters if no candidates found
        candidate_letters = set('ANIOERULGSHTMKCBDPYQZVJWFX') - set(guesses)
    
    # Calculate pure information gain for each candidate letter
    info_gains = {}
    for letter in candidate_letters:
        info_gains[letter] = calculate_information_gain(possible_words, probabilities, letter)
    
    # Calculate positional value (how well a letter discriminates at specific positions)
    positional_values = {}
    for letter in candidate_letters:
        # Track how well this letter splits the possible words at each position
        position_splits = {}
        for i in range(len(feedback)):
            if feedback[i] == '-':  # Only consider unrevealed positions
                # Count words with and without this letter at position i
                with_letter = sum(1 for word in possible_words if i < len(word) and word[i] == letter)
                total = len(possible_words)
                # Value is highest when split is close to 50/50
                if total > 0:
                    split_ratio = with_letter / total
                    # This formula peaks at 0.5 (perfect split)
                    position_splits[i] = 4 * split_ratio * (1 - split_ratio)
        
        # Use the best positional split value for this letter
        positional_values[letter] = max(position_splits.values()) if position_splits else 0
    
    # Combine the metrics with balanced weights
    # Start with base weights
    info_gain_weight = 0.7
    positional_weight = 0.3
    
    # Adjust weights based on game stage
    revealed_ratio = feedback.count('-') / len(feedback)
    if revealed_ratio < 0.5:  # Late game: favor positional discrimination
        info_gain_weight = 0.6
        positional_weight = 0.4
    
    # Calculate final scores
    letter_scores = {}
    for letter in candidate_letters:
        # Normalize values to prevent one metric from dominating
        norm_gain = info_gains[letter] / max(info_gains.values()) if max(info_gains.values()) > 0 else 0
        norm_pos = positional_values[letter] / max(positional_values.values()) if max(positional_values.values()) > 0 else 0
        
        letter_scores[letter] = (info_gain_weight * norm_gain + 
                               positional_weight * norm_pos)
    
    # Select the letter with the highest score
    best_letter = max(letter_scores.items(), key=lambda x: x[1])[0]
    
    return best_letter
#------------------------------------------------------------------------------------------------------
def calculate_positional_frequencies(possible_words, feedback):
    positional_frequencies = defaultdict(int)
    total_unrevealed = feedback.count('-')  # Total unrevealed positions
    
    for word in possible_words:
        for i, letter in enumerate(word):
            if feedback[i] == '-':  # Only consider unrevealed positions
                positional_frequencies[letter] += 1
    
    # Normalize by total unrevealed positions
    if total_unrevealed > 0:
        for letter in positional_frequencies:
            positional_frequencies[letter] /= total_unrevealed
    
    return positional_frequencies
#------------------------------------------------------------------------------------------------------
"""Main agent function"""
def agent_function(request_data, request_info):
    feedback = request_data['feedback']
    guesses = request_data['guesses']

    # If the word is completely revealed (no dashes) and not yet guessed as a whole word, return it immediately.
    if '-' not in feedback and feedback not in guesses:
        return feedback

    # Initialize word list if it's the first run.
    if not hasattr(agent_function, 'word_list'):
        agent_function.word_list = all_cities  # Assuming all_cities is your initial list
        agent_function.previous_feedback = ""
        agent_function.incorrect_words = set()

    # If feedback hasn't changed after a letter guess, filter out incompatible words.
    if len(guesses) > 0:
        last_guess = guesses[-1]
        if len(last_guess) == 1:  # Letter guess
            if agent_function.previous_feedback == feedback:
                # If feedback didn't change, the letter is not in the word
                agent_function.word_list = [word for word in agent_function.word_list if last_guess not in word]
            else:
                # If feedback changed, the letter is in the word
                # Filter words to include only those with the letter in the correct positions
                new_word_list = []
                for word in agent_function.word_list:
                    if last_guess in word:
                        # Check if the letter positions match the feedback
                        match = True
                        for i, (char, fb_char) in enumerate(zip(word, feedback)):
                            if fb_char != '-' and char != fb_char:
                                match = False
                                break
                        if match:
                            new_word_list.append(word)
                agent_function.word_list = new_word_list
        else:  # Word guess
            if last_guess not in agent_function.incorrect_words:
                agent_function.incorrect_words.add(last_guess)

    # Filter words based on feedback and previous guesses.
    possible_words = filter_words(agent_function.word_list, feedback, guesses)

    possible_words = [word for word in possible_words if word not in agent_function.incorrect_words]

    # If no words match, reset the word list and try again.
    if not possible_words:
        possible_words = [city for city in all_cities if len(city) == len(feedback)]
        possible_words = filter_words(possible_words, feedback, guesses)

    agent_function.word_list = possible_words  # Update word list after filtering
    probabilities = update_word_probabilities(possible_words)

    # If no letter guesses made yet, use the updated initial guess
    if len(guesses) == 0:
        return get_initial_guess(len(feedback), feedback, guesses)
    
    # If only a few words remain and we've made at least 3 letter guesses, try guessing the full word.
    if len(possible_words) <= 3 and sum(1 for g in guesses if len(g) == 1) >= 3:
        word_probabilities = update_word_probabilities(possible_words)
        sorted_words = [w for _, w in sorted(zip(word_probabilities, possible_words), reverse=True)]
        for word in sorted_words:
            if word not in agent_function.incorrect_words and word not in guesses:
                return word

    # If exactly one word remains, return it as the full word guess.
    if len(possible_words) == 1:
        word = possible_words[0]
        if word not in guesses:
            return word

    # Fallback: If no valid word is found, return the first letter from the alphabet that hasn't been guessed.
    if not possible_words:
        for letter in 'ANIOERULGSHTMKCBDPYQZVJWFX':
            if letter not in guesses:
                return letter

    # Otherwise, use information gain on candidate letters to find the best letter to guess.
    candidate_letters = set(letter for word in possible_words for letter in word) - set(guesses)
    if not candidate_letters:
        candidate_letters = set('ANIOERULGSHTMKCBDPYQZVJWFX') - set(guesses)
    
    # Cache candidate frequencies (compute once per turn)
    candidate_frequencies = {
        letter: sum(word.count(letter) for word in possible_words)
        for letter in candidate_letters
    }
    
    # Compute positional frequency for candidate letters in unrevealed positions.
    positional_frequencies = {letter: 0 for letter in candidate_letters}
    for word in possible_words:
        for i, letter in enumerate(word):
            # Only consider positions that are still unrevealed.
            if feedback[i] == '-' and letter in positional_frequencies:
                positional_frequencies[letter] += 1
    # Normalize the positional frequencies (avoid division by zero)
    max_pos_freq = max(positional_frequencies.values()) if positional_frequencies else 1

    # Adaptive hyperparameter for positional factor weight.
    # Initialize if not already set.
    if not hasattr(agent_function, 'adaptive_pos_weight'):
        agent_function.adaptive_pos_weight = 1.0

    # Calculate average unrevealed frequency across candidate letters.
    average_unrevealed = (sum(positional_frequencies.values()) / len(positional_frequencies)
                          if positional_frequencies else 0)

    # Define threshold values (experiment with these).
    THRESHOLD_HIGH = 2.0  # Example: high average unrevealed count.
    THRESHOLD_LOW  = 1.0  # Example: low average unrevealed count.

    # Adjust adaptive weight based on average unrevealed positions.
    if average_unrevealed > THRESHOLD_HIGH:
        agent_function.adaptive_pos_weight *= 1.1  # Increase weight if unrevealed positions remain high.
    elif average_unrevealed < THRESHOLD_LOW:
        agent_function.adaptive_pos_weight *= 0.9  # Decrease weight if few unrevealed positions.

    # Determine a trend multiplier.
    trend_multiplier = 1.0
    if agent_function.previous_feedback == feedback:
        trend_multiplier = 1.2

    best_letter = None
    max_weighted_gain = -1
    for letter, freq in candidate_frequencies.items():
        gain = calculate_information_gain(possible_words, probabilities, letter)
        # Normalize positional score between 0 and 1.
        pos_factor = positional_frequencies[letter] / max_pos_freq
        # Multiply the normalized positional factor with the adaptive hyperparameter.
        combined_pos_factor = (1 + pos_factor * agent_function.adaptive_pos_weight)
        if letter not in feedback:
            weighted_gain = gain * freq * combined_pos_factor * trend_multiplier
        else:
            weighted_gain = gain * freq * combined_pos_factor
        if weighted_gain > max_weighted_gain:
            max_weighted_gain = weighted_gain
            best_letter = letter

    best_letter = select_best_letter(possible_words, probabilities, feedback, guesses)
    return best_letter

#------------------------------------------------------------------------------------------------------
"""Run the agent"""
if __name__ == '__main__':
    import sys, logging
    from client import run

    # You can set the logging level to logging.WARNING or logging.ERROR for less output.
    logging.basicConfig(level=logging.INFO)

    run(
        agent_config_file=sys.argv[1],
        agent=agent_function,
        parallel_runs=True,     # Set it to False for debugging.
        run_limit=100000000,         # Stop after 100000000 runs. Set to 1 for debugging.
    )