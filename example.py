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

# Cache for initial guesses based on word length
initial_guess_cache = {}
#------------------------------------------------------------------------------------------------------
"""Simulate the biased selection (50% Tanzania, 50% other)"""
def get_weighted_cities():
    # Create weights based on the 50/50 bias
    weights = []
    for city in all_cities:
        if city in tanzanian_cities:
            weights.append(0.5 / len(tanzanian_cities))
        else:
            weights.append(0.5 / len(non_tanzanian_cities))
    
    return all_cities, weights
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
# Helper function to check if a word is compatible with the feedback
def is_compatible(word, feedback):
    if len(word) > len(feedback):
        return False
        
    for i, (w_char, f_char) in enumerate(zip(word, feedback)):
        if f_char != '-' and w_char != f_char:
            return False
    
    return True
#------------------------------------------------------------------------------------------------------
"""Filter the word list based on feedback and previous guesses."""
def filter_words(word_list, feedback, guesses):
    filtered_list = []
    
    for word in word_list:
        if len(word) != len(feedback):
            continue
        
        is_match = True
        for i, (char, feedback_char) in enumerate(zip(word, feedback)):
            # If the feedback shows a letter at this position
            if feedback_char != '-':
                if char != feedback_char:
                    is_match = False
                    break
                
            # If the feedback shows a dash at this position
            else:
                # If we've guessed this letter before and it's not revealed here
                if char in guesses and char != feedback_char:
                    is_match = False
                    break
        
        # Check if the word contains any letters that were guessed but not revealed
        for guess in guesses:
            if len(guess) == 1:  # It's a letter guess
                if guess in word and guess not in feedback:
                    is_match = False
                    break
        
        if is_match:
            filtered_list.append(word)
    
    return filtered_list

#------------------------------------------------------------------------------------------------------
"""Filter the word list based on feedback and previous guesses for advanced rules."""
def filter_words_advanced(word_list, feedback, guesses):
    filtered_list = []
    
    for word in word_list:
        # In advanced rules, word length is disguised, so we need to check if the word is compatible
        if len(word) > len(feedback):
            continue
        
        # Check if the word matches the feedback pattern
        is_match = True
        letter_positions = {}
        
        # First, check if revealed letters match
        for i, feedback_char in enumerate(feedback):
            if feedback_char != '-':
                # If we've gone beyond the word length or the letter doesn't match
                if i >= len(word) or word[i] != feedback_char:
                    is_match = False
                    break
                
                # Keep track of revealed positions for each letter
                if feedback_char not in letter_positions:
                    letter_positions[feedback_char] = []
                letter_positions[feedback_char].append(i)
        
        if not is_match:
            continue
        
        # Check if the word contains guessed letters that weren't revealed
        for guess in guesses:
            if len(guess) == 1:  # It's a letter guess
                occurrences_in_word = [i for i, char in enumerate(word) if char == guess]
                
                # If the letter is in the word but not in the feedback at all
                if occurrences_in_word and guess not in feedback:
                    is_match = False
                    break
                
                # If the letter is in the feedback but not at all expected positions
                if guess in letter_positions:
                    revealed_positions = letter_positions[guess]
                    # There should be at least as many occurrences in the word as revealed
                    if len(occurrences_in_word) < len(revealed_positions):
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
    
    # Calculate probabilities
    for word in word_list:
        if word in tanzanian_cities:
            # 50% chance of being a Tanzanian city
            prob = 0.5 / len(tz_cities) if len(tz_cities) > 0 else 0
        else:
            # 50% chance of being a non-Tanzanian city
            prob = 0.5 / len(non_tz_cities) if len(non_tz_cities) > 0 else 0
        probabilities.append(prob)
    
    return probabilities

#------------------------------------------------------------------------------------------------------
"""Get the best initial guess for a given word length."""
def get_initial_guess(word_length):
    if word_length in initial_guess_cache:
        return initial_guess_cache[word_length]
    
    # Filter cities by length
    possible_words = [city for city in all_cities if len(city) == word_length]
    
    # Calculate probabilities
    probabilities = update_word_probabilities(possible_words)
    
    # Find the letter with the highest information gain
    best_letter = None
    max_gain = -1
    for letter in string.ascii_uppercase:
        gain = calculate_information_gain(possible_words, probabilities, letter)
        if gain > max_gain:
            max_gain = gain
            best_letter = letter
    
    # Cache the result
    initial_guess_cache[word_length] = best_letter
    return best_letter

#------------------------------------------------------------------------------------------------------
"""Main agent function"""
def agent_function(request_data, request_info):
    feedback = request_data['feedback']
    guesses = request_data['guesses']
    
    # print(f"Current feedback: {feedback}")
    # print(f"Previous guesses: {guesses}")
    # print("--------------------------------------------------")
    
    # If the word is completely revealed (no dashes) and not yet guessed as a whole word
    if '-' not in feedback and feedback not in guesses:
        return feedback
    
    # Initialize the word list and previous feedback if this is the first guess
    if not hasattr(agent_function, 'word_list'):
        agent_function.word_list = all_cities
        agent_function.advanced_rules = len(feedback) > 7  # Guess if advanced rules are used
        agent_function.previous_feedback = ""
        agent_function.incorrect_words = set()  # Track incorrect word guesses
    
    # If the feedback hasn't changed after a letter guess, that letter is not in the word
    if agent_function.previous_feedback == feedback and len(guesses) > 0:
        last_guess = guesses[-1]
        if len(last_guess) == 1:  # It's a letter guess
            # Further filter words that contain this letter
            agent_function.word_list = [word for word in agent_function.word_list if last_guess not in word]
        elif last_guess not in agent_function.incorrect_words:
            # If a word guess was incorrect, add it to our set of incorrect word guesses
            agent_function.incorrect_words.add(last_guess)
    
    # Update previous feedback
    agent_function.previous_feedback = feedback
    
    # Determine if we're using standard or advanced rules
    if agent_function.advanced_rules:
        possible_words = filter_words_advanced(agent_function.word_list, feedback, guesses)
    else:
        possible_words = filter_words(agent_function.word_list, feedback, guesses)
    
    # Filter out words that we've already guessed and were incorrect
    possible_words = [word for word in possible_words if word not in agent_function.incorrect_words]
    
    # If no words match the criteria, reset the word list and try again
    if not possible_words:
        # Reset the word list to all cities that match the current feedback pattern
        if agent_function.advanced_rules:
            possible_words = [city for city in all_cities if len(city) <= len(feedback)]
        else:
            possible_words = [city for city in all_cities if len(city) == len(feedback)]
        
        # Further filter using the current feedback
        possible_words = [word for word in possible_words if is_compatible(word, feedback)]
        # Remove any words we've already tried
        possible_words = [word for word in possible_words if word not in agent_function.incorrect_words]
    
    agent_function.word_list = possible_words
    
    # Calculate probabilities for possible_words
    probabilities = update_word_probabilities(possible_words)
    
    # If few words remain AND we've made at least 3 letter guesses, we can guess the word
    if len(possible_words) <= 3 and sum(1 for g in guesses if len(g) == 1) >= 3:
        # Sort by probability if using Tanzanian bias
        word_probabilities = update_word_probabilities(possible_words)
        if word_probabilities:
            sorted_words = [w for _, w in sorted(zip(word_probabilities, possible_words), reverse=True)]
            # Return the most likely word that hasn't been guessed yet
            for word in sorted_words:
                if word not in agent_function.incorrect_words and word not in guesses:
                    return word
    
    # Otherwise, find the best letter to guess
    best_letter = None
    max_gain = -1
    for letter in string.ascii_uppercase:
        if letter in guesses:
            continue
        gain = calculate_information_gain(possible_words, probabilities, letter)
        if gain > max_gain:
            max_gain = gain
            best_letter = letter
    
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
