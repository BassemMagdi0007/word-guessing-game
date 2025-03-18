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
cities_df['city_ascii'] = cities_df['city_ascii'].str.upper()

# Separate Tanzanian and non-Tanzanian cities
tanzanian_cities = cities_df[cities_df['iso2'] == 'TZ']['city_ascii'].tolist()
non_tanzanian_cities = cities_df[cities_df['iso2'] != 'TZ']['city_ascii'].tolist()
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
"""Filter the word list based on feedback and previous guesses for advanced rules."""
def filter_words_advanced(word_list, feedback, guesses):

    # Apply logic here
    
    return
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
"""Select the best letter to guess based on information gain and positional value."""
def select_best_letter(possible_words, probabilities, feedback, guesses):
    # Get all unique letters from possible words that haven't been guessed yet
    candidate_letters = set(letter for word in possible_words for letter in word) - set(guesses)
    
    # If no candidate letters are found, fall back to the entire alphabet (excluding guessed letters)
    if not candidate_letters:
        candidate_letters = set(string.ascii_uppercase) - set(guesses)
    
    # If still no candidates, return None (this should not happen, but it's a safeguard)
    if not candidate_letters:
        return None
    
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
    if letter_scores:
        best_letter = max(letter_scores.items(), key=lambda x: x[1])[0]
        return best_letter
    else:
        # Fallback: return the first unguessed letter from the alphabet
        return next((letter for letter in string.ascii_uppercase if letter not in guesses), None)

#------------------------------------------------------------------------------------------------------
"""Main agent function"""
def agent_function(request_data, request_info):
    feedback = request_data['feedback']
    guesses = request_data['guesses']

    # Initialize word list and set advanced rules flag if it's the first run.
    if not hasattr(agent_function, 'word_list'):
        agent_function.word_list = all_cities  # Assuming all_cities is your initial list
        agent_function.incorrect_words = set()

    # Filter possible words based on feedback and guesses
    possible_words = filter_words(agent_function.word_list, feedback, guesses)
    possible_words = [word for word in possible_words 
                      if word not in agent_function.incorrect_words]

    # Fallback if filtering eliminates all words
    if not possible_words:
        possible_words = [city for city in all_cities if len(city) == len(feedback)]
        possible_words = filter_words(possible_words, feedback, guesses)

    # Update word list and probabilities
    agent_function.word_list = possible_words
    probabilities = update_word_probabilities(possible_words)

    # Early word guessing conditions
    if (len(possible_words) <= 3 and sum(1 for g in guesses if len(g) == 1) >= 2) \
       or (probabilities and max(probabilities) > 0.7):
        word_prob_pairs = sorted(zip(probabilities, possible_words), reverse=True)
        for prob, word in word_prob_pairs:
            if word not in agent_function.incorrect_words and word not in guesses:
                return word

    # Single word remaining
    if len(possible_words) == 1:
        word = possible_words[0]
        if word not in guesses:
            return word

    # Select best letter using optimized scoring
    best_letter = select_best_letter(possible_words, probabilities, feedback, guesses)
    if best_letter:
        return best_letter

    # Final fallback: return the first unguessed letter from the alphabet
    for letter in 'ANIOERULGSHTMKCBDPYQZVJWFX':
        if letter not in guesses:
            return letter

    # Final fallback (should never reach here)
    return "A"

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