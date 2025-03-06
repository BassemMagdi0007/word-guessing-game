"""
    python example.py env/simple-env.json
"""
import pandas as pd
import string
import random
import math
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
def calculate_information_gain(cities, letter):
    total_cities = len(cities)
    if total_cities <= 1:
        return 0
    
    # Current entropy (before guessing)
    current_entropy = -sum(1/total_cities * math.log2(1/total_cities) for _ in range(total_cities))
    
    # Group cities by the pattern they would create if we guessed the letter
    patterns = {}
    for city in cities:
        pattern = ""
        for char in city:
            if char == letter:
                pattern += letter
            else:
                pattern += "-"
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(city)
    
    # Calculate entropy after guessing (weighted by probability of each pattern)
    new_entropy = 0
    for pattern, matching_cities in patterns.items():
        probability = len(matching_cities) / total_cities
        if probability > 0:
            entropy = -sum(1/len(matching_cities) * math.log2(1/len(matching_cities)) for _ in range(len(matching_cities)))
            new_entropy += probability * entropy
    
    # Information gain is the reduction in entropy
    return current_entropy - new_entropy
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
"""Main agent function"""
def agent_function(request_data, request_info):
    feedback = request_data['feedback']
    guesses = request_data['guesses']
    
    # Initialize the word list and previous feedback if this is the first guess
    if not hasattr(agent_function, 'word_list'):
        agent_function.word_list = all_cities
        agent_function.advanced_rules = len(feedback) > 7  # Guess if advanced rules are used
        agent_function.previous_feedback = ""
    
    # If the feedback hasn't changed after a letter guess, that letter is not in the word
    if agent_function.previous_feedback == feedback and len(guesses) > 0:
        last_guess = guesses[-1]
        if len(last_guess) == 1:  # It's a letter guess
            # Further filter words that contain this letter
            agent_function.word_list = [word for word in agent_function.word_list if last_guess not in word]
    
    # Update previous feedback
    agent_function.previous_feedback = feedback
    
    # Determine if we're using standard or advanced rules
    if agent_function.advanced_rules:
        possible_words = filter_words_advanced(agent_function.word_list, feedback, guesses)
    else:
        possible_words = filter_words(agent_function.word_list, feedback, guesses)
    
    # If no words match the criteria, reset the word list and try again
    # This is a recovery mechanism, not a fallback
    if not possible_words:
        # Reset the word list to all cities that match the current feedback pattern
        if agent_function.advanced_rules:
            possible_words = [city for city in all_cities if len(city) <= len(feedback)]
        else:
            possible_words = [city for city in all_cities if len(city) == len(feedback)]
        
        # Further filter using the current feedback
        possible_words = [word for word in possible_words if is_compatible(word, feedback)]
    
    agent_function.word_list = possible_words
    
    # If only one word remains, guess it
    if len(possible_words) == 1:
        return possible_words[0]
    
    # If few words remain, might be better to guess the word directly
    if len(possible_words) <= 3:
        # Find a word that hasn't been guessed yet
        for word in possible_words:
            if word not in guesses:
                return word
    
    # Otherwise, find the best letter to guess
    best_letter = None
    max_gain = -1
    
    for letter in string.ascii_uppercase:
        if letter in guesses:
            continue
        
        gain = calculate_information_gain(possible_words, letter)
        if gain > max_gain:
            max_gain = gain
            best_letter = letter
    
    # If no letter found, find any unguessed letter
    if best_letter is None:
        for letter in string.ascii_uppercase:
            if letter not in guesses:
                return letter
        
    return best_letter

# Helper function to check if a word is compatible with the feedback
def is_compatible(word, feedback):
    if len(word) > len(feedback):
        return False
        
    for i, (w_char, f_char) in enumerate(zip(word, feedback)):
        if f_char != '-' and w_char != f_char:
            return False
    
    return True
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
        run_limit=1000,         # Stop after 1000 runs. Set to 1 for debugging.
    )

