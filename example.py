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
def calculate_information_gain(possible_words, probabilities, letter, advanced_rules=False, exhausted_letters=None):
    if len(possible_words) == 0:
        return 0
        
    # If this letter has been guessed multiple times with no change in feedback
    if advanced_rules and exhausted_letters and letter in exhausted_letters:
        return 0  # No information gain for exhausted letters

    # Current entropy (using actual probabilities)
    current_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    # Handle standard rules
    if not advanced_rules:
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

    # Handle advanced rules
    else:
        # Group words by possible outcomes of guessing this letter
        # In advanced rules, only one occurrence is revealed at a time (if any)
        outcome_probs = defaultdict(float)
        outcome_words = defaultdict(list)
        
        for word, prob in zip(possible_words, probabilities):
            # Count occurrences of letter in the word
            occurrences = [(i, char) for i, char in enumerate(word) if char == letter]
            
            if not occurrences:
                # Letter doesn't occur
                outcome = "none"
                outcome_probs[outcome] += prob
                outcome_words[outcome].append((word, prob))
            else:
                # Letter occurs once or more
                # In advanced rules, each position is equally likely to be revealed
                for pos, _ in occurrences:
                    outcome = f"pos_{pos}"
                    # Divide probability by number of occurrences since any could be revealed
                    outcome_probs[outcome] += prob / len(occurrences)
                    outcome_words[outcome].append((word, prob / len(occurrences)))
        
        # Calculate expected entropy after guessing the letter
        expected_entropy = 0
        for outcome, total_prob in outcome_probs.items():
            if total_prob == 0:
                continue
            # Calculate entropy for this outcome
            conditional_probs = [prob / total_prob for (_, prob) in outcome_words[outcome]]
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
    filtered_list = []
    letter_guesses = [g for g in guesses if len(g) == 1]
    
    # In advanced rules, the actual word length is unknown and could be shorter than the feedback
    # We need to consider all possible word lengths up to the feedback length
    max_possible_length = len(feedback)
    
    for word in word_list:
        # Skip words longer than the feedback
        if len(word) > max_possible_length:
            continue
            
        is_match = True
        
        # Check if the word is compatible with revealed positions
        for i in range(len(word)):
            if i < len(feedback) and feedback[i] != '-' and word[i] != feedback[i]:
                is_match = False
                break
        
        if not is_match:
            continue
        
        # For each guessed letter that appears in the word
        for letter in letter_guesses:
            # Count occurrences in the word
            word_occurrences = word.count(letter)
            
            # Count revealed occurrences in the feedback
            feedback_occurrences = feedback.count(letter)
            
            # If we've guessed this letter but not all occurrences are revealed yet
            if letter in word and letter in guesses:
                # The number of revealed occurrences should be less than or equal to 
                # the actual occurrences in the word
                if feedback_occurrences > word_occurrences:
                    is_match = False
                    break
            
            # If we've guessed this letter but it doesn't appear in the feedback at all
            # the word shouldn't contain this letter
            if letter in guesses and letter not in feedback and letter in word:
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
"""Select the best letter to guess based on information gain and positional value."""
def select_best_letter(possible_words, probabilities, feedback, guesses, advanced_rules=False, exhausted_letters=None):
    # Get all unique letters from possible words that haven't been guessed yet
    candidate_letters = set(letter for word in possible_words for letter in word)
    
    if advanced_rules:
        # In advanced rules, we might want to guess a letter again if it might have more occurrences
        guessed_letters = [g for g in guesses if len(g) == 1]
        
        # Calculate expected occurrences of each letter
        expected_occurrences = {}
        for letter in set("".join(possible_words)):
            expected_occurrences[letter] = sum(word.count(letter) * prob for word, prob in zip(possible_words, probabilities))
        
        # For already guessed letters, only keep them as candidates if likely more occurrences exist
        for letter in guessed_letters:
            revealed_count = feedback.count(letter)
            
            # Never consider letters that have been guessed twice with no change in feedback
            if exhausted_letters and letter in exhausted_letters:
                if letter in candidate_letters:
                    candidate_letters.remove(letter)
                continue
                
            if letter in expected_occurrences and expected_occurrences[letter] > revealed_count + 0.5:
                # Keep this letter as a candidate
                pass
            else:
                # Remove this letter from candidates
                if letter in candidate_letters:
                    candidate_letters.remove(letter)
    else:
        # In standard rules, remove all previously guessed letters
        candidate_letters -= set(guesses)
    
    # If no candidate letters are found, fall back to the entire alphabet (excluding fully revealed letters)
    if not candidate_letters:
        if advanced_rules:
            # In advanced rules, fully excluded letters are those that don't appear in feedback
            # when we've guessed them, or letters that have been guessed twice with no change
            excluded_letters = {g for g in guesses if len(g) == 1 and g not in feedback}
            if exhausted_letters:
                excluded_letters.update(exhausted_letters)
            candidate_letters = set(string.ascii_uppercase) - excluded_letters
        else:
            candidate_letters = set(string.ascii_uppercase) - set(guesses)
    
    # If still no candidates, return None (this should not happen, but it's a safeguard)
    if not candidate_letters:
        return None
    
    # Calculate pure information gain for each candidate letter
    info_gains = {}
    for letter in candidate_letters:
        info_gains[letter] = calculate_information_gain(possible_words, probabilities, letter, advanced_rules, exhausted_letters)
    
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
    
    # Prioritize letters with high expected occurrences in advanced rules
    if advanced_rules:
        expected_letter_counts = {}
        for letter in candidate_letters:
            expected_letter_counts[letter] = sum(word.count(letter) * prob for word, prob in zip(possible_words, probabilities))
            
        for letter in letter_scores:
            if letter in expected_letter_counts:
                # Boost score for letters with many expected occurrences
                letter_scores[letter] *= (1 + 0.2 * min(expected_letter_counts[letter], 3))
    
    # Apply a strong penalty to letters that have been consecutively repeated with no change
    for letter in letter_scores:
        if letter in agent_function.consecutive_repeats and agent_function.consecutive_repeats[letter] > 0:
            letter_scores[letter] *= 0.1  # Apply a 90% penalty
    
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

    # Initialize state variables if first run
    if not hasattr(agent_function, 'word_list'):
        agent_function.word_list = all_cities
        agent_function.incorrect_words = set()
        agent_function.advanced_rules = None
        agent_function.letter_counts = defaultdict(int)  # Track guessed letters and their appearances
        agent_function.letter_feedback_history = {}  # Track feedback after each letter guess
        agent_function.consecutive_repeats = defaultdict(int)  # Track consecutive repeats of a letter
        agent_function.exhausted_letters = set()  # Track letters that have been guessed twice with no change in feedback

    # Detect if we're using advanced rules by checking if the same letter appears multiple times in guesses
    if agent_function.advanced_rules is None:
        letter_guesses = [g for g in guesses if len(g) == 1]
        duplicate_letters = len(letter_guesses) != len(set(letter_guesses))
        # Also check if a letter appears in guesses but not all occurrences are revealed
        letter_mismatch = False
        for letter in letter_guesses:
            if letter in feedback:
                for word in agent_function.word_list:
                    if word.count(letter) > feedback.count(letter):
                        letter_mismatch = True
                        break
        agent_function.advanced_rules = duplicate_letters or letter_mismatch

    # Track feedback history for each letter to detect when a letter stops revealing new positions
    if len(guesses) > 0 and len(guesses[-1]) == 1:
        last_letter = guesses[-1]
        
        # Store the feedback after this letter guess
        if last_letter not in agent_function.letter_feedback_history:
            agent_function.letter_feedback_history[last_letter] = []
        agent_function.letter_feedback_history[last_letter].append(feedback)
        
        # Check if feedback changed since the last guess of this letter
        if len(agent_function.letter_feedback_history[last_letter]) >= 2:
            if agent_function.letter_feedback_history[last_letter][-1] == agent_function.letter_feedback_history[last_letter][-2]:
                agent_function.consecutive_repeats[last_letter] += 1
                # Immediately mark letter as exhausted if guessed with no change
                agent_function.exhausted_letters.add(last_letter)
            else:
                # Reset counter if we found a new position
                agent_function.consecutive_repeats[last_letter] = 0

    # Filter possible words based on feedback and guesses
    if agent_function.advanced_rules:
        possible_words = filter_words_advanced(agent_function.word_list, feedback, guesses)
    else:
        possible_words = filter_words(agent_function.word_list, feedback, guesses)
    
    possible_words = [word for word in possible_words 
                      if word not in agent_function.incorrect_words]

    # Fallback if filtering eliminates all words
    if not possible_words:
        if agent_function.advanced_rules:
            possible_words = [city for city in all_cities if len(city) <= len(feedback)]
            possible_words = filter_words_advanced(possible_words, feedback, guesses)
        else:
            possible_words = [city for city in all_cities if len(city) == len(feedback)]
            possible_words = filter_words(possible_words, feedback, guesses)

    # Update word list and probabilities
    agent_function.word_list = possible_words
    probabilities = update_word_probabilities(possible_words)

    # For advanced rules, check if we need to repeat a letter guess
    if agent_function.advanced_rules:
        # Calculate expected number of occurrences for each guessed letter
        expected_occurrences = {}
        for letter in set(l for l in string.ascii_uppercase if l in "".join(possible_words)):
            expected_occurrences[letter] = sum(word.count(letter) * prob for word, prob in zip(possible_words, probabilities))
        
        # Check if any previously guessed letter likely has more unrevealed occurrences
        for letter in [g for g in guesses if len(g) == 1]:
            # Never repeat exhausted letters
            if letter in agent_function.exhausted_letters:
                continue
                
            revealed_count = feedback.count(letter)
            
            # Only consider repeating a letter if:
            # 1. Expected occurrences is significantly higher than revealed count
            # 2. The letter has NEVER been repeated without changes
            if (letter in expected_occurrences and 
                expected_occurrences[letter] > revealed_count + 0.5 and
                agent_function.consecutive_repeats.get(letter, 0) == 0):
                # There's likely more occurrences to be found
                return letter

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
    best_letter = select_best_letter(
        possible_words, 
        probabilities, 
        feedback, 
        guesses, 
        agent_function.advanced_rules,
        agent_function.exhausted_letters
    )
    if best_letter:
        return best_letter

    # Final fallback: return common letters in English
    for letter in 'ANIOERULGSHTMKCBDPYQZVJWFX':
        if (letter not in guesses or agent_function.advanced_rules) and letter not in agent_function.exhausted_letters:
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