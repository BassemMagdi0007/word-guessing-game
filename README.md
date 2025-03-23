# Word-Guessing Game

## Table of Contents

- [Introduction](#introduction)
  - [Key Features](#key-features)
- [Setup](#setup)
  - [Repository Content](#repository-content)
  - [How to Run the Code](#how-to-run-the-code)
  - [Used Libraries](#used-libraries)
- [Code Structure](#code-structure)
- [Self-Evaluation and Design Decisions](#self-evaluation-and-design-decisions)
- [Output Format](#output-format)


## Introduction

This project develops an AI agent for the 'Guess the Word' game within a structured environment, where the agent must deduce a hidden word based on feedback from previous guesses. The agent leverages probabilistic reasoning and reinforcement learning to refine its guessing strategy, dynamically adjusting its approach based on observed patterns. By optimizing its decision-making process, the agent aims to minimize the number of guesses required while balancing exploration and exploitation. The implementation integrates an adaptive learning mechanism to enhance word prediction accuracy, ensuring an efficient and intelligent gameplay experience.

### Key Features
- **Probabilistic Reasoning**: The agent uses probability distributions to prioritize likely cities.
- **Information Gain**: The agent selects letters that maximize the reduction in uncertainty (entropy).
- **Adaptive Learning**: The agent dynamically adjusts its strategy based on feedback and game rules.
- **Tanzanian Bias**: The agent prioritizes Tanzanian cities, introducing a regional bias in its guessing logic.


## Setup

### Repository Content
The repository contains the following files:
- **example.py**: The main code for the AI agent.
- **filtered_worldcities.csv**: The filtered dataset of world cities used by the agent.
- **filterCSV.py**: The script used to filter the original `worldcities.csv` file to create `filtered_worldcities.csv`.

### How to Run the Code
To run the AI agent in different environments, use the following commands:
- **Simple Environment**:
  ```bash
  python example.py env/simple-env.json
  ```
- **Advanced Environment**:
  ```bash
  python example.py env/advanced-env.json
  ```

### Used Libraries
The following Python libraries are used in this project:
- **pandas**: For data manipulation and preprocessing.
- **math**: For mathematical operations, such as calculating entropy.
- **collections.defaultdict**: For efficient storage and retrieval of data.
- **string**: For handling string operations, such as uppercase conversion.

## Code Structure

The code is organized into the following key components:
## 1. **Data Preprocessing**
```python
# Load the CSV file containing world cities data
cities_df = pd.read_csv('worldcities.csv')

# Keep only cities with a population of at least 100,000
cities_df = cities_df[cities_df['population'] >= 100000]

# Remove city names that contain diacritics, hyphens, or spaces
# This ensures the original city name matches its ASCII equivalent (without accents) and contains no spaces or hyphens
cities_df = cities_df[cities_df['city_ascii'].apply(lambda x: x == unidecode(x) and ' ' not in x and '-' not in x)]

# Further filter out city names that contain any non-alphabetic characters (allowing only A-Z letters)
cities_df = cities_df[cities_df['city_ascii'].str.match(r'^[A-Za-z]+$')]

# Save the cleaned dataset to a new CSV file
cities_df.to_csv('filtered_worldcities.csv', index=False)

print("Filtered CSV file saved as 'filtered_worldcities.csv'")
```

### **Preprocessing Overview**  
- The script loads the `worldcities.csv` file and filters cities with a population of at least 100,000.  
- It removes city names containing diacritics, hyphens, spaces, or non-alphabetic characters to ensure uniformity.  
- The cleaned dataset is saved as `filtered_worldcities.csv` for further use in the AI agent.  

```python
# Load the pre-filtered city data
cities_df = pd.read_csv('filtered_worldcities.csv')

# Convert city names to uppercase for consistency
cities_df['city_ascii'] = cities_df['city_ascii'].str.upper()

# Separate Tanzanian and non-Tanzanian cities
tanzanian_cities = cities_df[cities_df['iso2'] == 'TZ']['city_ascii'].tolist()
non_tanzanian_cities = cities_df[cities_df['iso2'] != 'TZ']['city_ascii'].tolist()
all_cities = tanzanian_cities + non_tanzanian_cities
```

### **Purpose:**  
Prepares the dataset of world cities for use in the AI agent by filtering, cleaning, and structuring the data for efficient and accurate word predictions.  

### **Details:**  
- The original dataset, `worldcities.csv`, is filtered to retain only cities with a population of at least 100,000.  
- Cities with diacritics, hyphens, spaces, or non-alphabetic characters in their names are removed using the `unidecode` library and regex filtering.  
- The cleaned dataset is saved as `filtered_worldcities.csv` for further processing.  
- City names are standardized by converting them to uppercase to ensure consistency during comparisons.  
- Cities are categorized into Tanzanian and non-Tanzanian groups based on their ISO2 country codes.  
- A combined list of all cities is created, with Tanzanian cities prioritized to introduce a regional bias in the agent’s guessing logic.  

### **Important Notes:**  
- The Tanzanian bias is implemented by prioritizing cities from Tanzania during probability updates.  
- The dataset must contain the `city_ascii` and `iso2` columns for this preprocessing to work.  


## 2. **Information Gain Calculation**

```python
def calculate_information_gain(possible_words, probabilities, letter, advanced_rules=False, exhausted_letters=None):
    # ...
```

### **Purpose and Theoretical Background**  
The `calculate_information_gain` function is rooted in **information theory**, specifically the concept of **entropy**. Entropy measures the uncertainty or unpredictability of a system. In this context, the system is the set of possible cities, and the goal is to reduce entropy by guessing letters that provide the most information.  

The function calculates the **expected reduction in entropy** (information gain) when a specific letter is guessed. This is achieved by comparing the **current entropy** of the system (before the guess) with the **expected entropy** after the guess.  

### **Mathematical Formulation**  
1. **Current Entropy**:  
   The entropy of the current system is calculated using the formula:  
   ```python
   current_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
   ```  
   Here, `p` represents the probability of each city in the `possible_words` list.  

2. **Expected Entropy**:  
   The function simulates the possible outcomes of guessing a letter. For each outcome (e.g., the letter appearing in specific positions or not appearing at all), it calculates the conditional entropy. The expected entropy is the weighted sum of these conditional entropies:  
   ```python
   expected_entropy = sum(total_prob * entropy for outcome, total_prob in outcome_probs.items())
   ```  

3. **Information Gain**:  
   The information gain is the difference between the current entropy and the expected entropy:  
   ```python
   information_gain = current_entropy - expected_entropy
   ```  

### **Advanced Rules**  
When `advanced_rules` is enabled, the function accounts for scenarios where the same letter may appear multiple times in a city name. For example, if the letter `A` appears twice in a city name, the function calculates the probability of each occurrence separately. This is done by grouping words based on the positions of the guessed letter:  
```python
outcome = f"pos_{pos}"
outcome_probs[outcome] += prob / total_occurrences
```  

### **Exhausted Letters**  
Letters that have been guessed multiple times with no change in feedback are considered **exhausted**. These letters provide no additional information and are ignored:  
```python
if advanced_rules and exhausted_letters and letter in exhausted_letters:
    return 0
```  

### **Key Insights**  
- The function prioritizes letters that maximize information gain, ensuring efficient reduction of uncertainty.  
- Advanced rules allow the agent to handle complex scenarios, such as repeated letter occurrences.  
- Exhausted letters are dynamically excluded to avoid redundant guesses.  


## 3. **Word Filtering**

```python
def filter_words(word_list, feedback, guesses):
    # ...
```

### **Purpose and Theoretical Background**  
The `filter_words` function is designed for the **simple environment**, where each letter guess reveals **all occurrences** of that letter in the hidden word. It ensures that only cities consistent with the feedback and previous guesses are retained. This is critical for reducing the search space and improving the agent's efficiency.  

### **Constraint Checks**  
The function applies the following constraints to each city in the `word_list`:  
1. **Length Constraint**:  
   The city's length must match the feedback length:  
   ```python
   if len(word) != len(feedback):
       continue
   ```  

2. **Revealed Positions**:  
   The revealed letters in the feedback must match the corresponding positions in the city name:  
   ```python
   for i, (char, feedback_char) in enumerate(zip(word, feedback)):
       if feedback_char != '-' and char != feedback_char:
           is_match = False
           break
   ```  

3. **Guessed Letters**:  
   The city must not contain any guessed letters that are not in the feedback:  
   ```python
   for letter in letter_guesses:
       if letter in word and letter not in feedback:
           is_match = False
           break
   ```  

### **Example**  
From the assignment sheet:  
- **Guess: I**  
  - **Feedback**: `----I`  
  - **Valid City**: "MOSHI"  
  The function ensures that the city "MOSHI" matches the feedback `----I` and does not contain any guessed letters that are not in the feedback.  

- **Guess: H**  
  - **Feedback**: `-----`  
  - **Guess: I**  
  - **Feedback**: `-----`  
  - **Guess: E**  
  - **Feedback**: `--E--`  
  - **Guess: B**  
  - **Feedback**: `-BE--`  
  - **Valid City**: "MBEYA"  
  The function ensures that the city "MBEYA" matches the feedback `-BE--` and does not contain any guessed letters (`H`, `I`) that are not in the feedback.  

### **Key Insights**  
- The function ensures that only valid candidates are considered, reducing the computational complexity of subsequent steps.  
- It dynamically adapts to the feedback and guesses, ensuring consistency with the observed data.  


## 4. **Advanced Word Filtering**

```python
def filter_words_advanced(word_list, feedback, guesses):
    # ...
```

### **Purpose and Theoretical Background**  
The `filter_words_advanced` function is designed for the **advanced environment**, where the same letter may appear multiple times in the hidden word, and each guess reveals only **one occurrence** at a time. This function extends the basic filtering mechanism to handle such scenarios, ensuring that the agent can adapt to more complex feedback patterns.  

### **Key Differences from Basic Filtering**  
1. **Length Flexibility**:  
   Cities shorter than the feedback length are considered if they match the revealed letters:  
   ```python
   if len(word) > len(feedback):
       continue
   ```  

2. **Repeated Letter Handling**:  
   The function checks if the city contains at least as many occurrences of a letter as revealed in the feedback:  
   ```python
   for i in range(len(word)):
       if i < len(feedback) and feedback[i] != '-' and word[i] != feedback[i]:
           is_match = False
           break
   ```  

### **Example**  
From the assignment sheet:  
- **Guess: A**  
  - **Feedback**: `A-------------------`  
- **Guess: R**  
  - **Feedback**: `AR------------------`  
- **Guess: U**  
  - **Feedback**: `ARU-----------------`  
- **Guess: ARUSHA**  
  - **Feedback**: `ARUSHA`  
  The function ensures that the city "ARUSHA" matches the feedback `ARUSHA` and does not contain any guessed letters that are not in the feedback. The feedback may include extra dashes to disguise the word's length, and the function handles this by considering cities of varying lengths.  

### **Key Insights**  
- The function supports more flexible guessing strategies, accommodating complex feedback patterns.  
- It ensures that the agent can handle repeated letter occurrences, improving its adaptability.  


## 5. **Probability Updates**

```python
def update_word_probabilities(word_list):
    # ...
```

### **Purpose and Theoretical Background**  
The `update_word_probabilities` function implements a **Bayesian updating** mechanism. It adjusts the probabilities of each city based on the **Tanzanian bias**, ensuring that Tanzanian cities are prioritized during guessing. This function is used in both simple and advanced environments.  

### **Mathematical Formulation**  
1. **Tanzanian Bias**:  
   Tanzanian cities are assigned a higher weight (0.7 by default), while non-Tanzanian cities share the remaining probability mass (0.3):  
   ```python
   tz_weight = 0.7 if len(tz_cities) > 0 else 0
   ```  

2. **Probability Calculation**:  
   The probability of each city is calculated as:  
   ```python
   if word in tanzanian_cities:
       prob = tz_weight / len(tz_cities)
   else:
       prob = (1 - tz_weight) / len(non_tz_cities)
   ```  

### **Example**  
If there are 3 Tanzanian cities and 7 non-Tanzanian cities, the probabilities would be:  
- Tanzanian cities: `0.7 / 3 ≈ 0.233` each.  
- Non-Tanzanian cities: `0.3 / 7 ≈ 0.043` each.  

### **Key Insights**  
- The Tanzanian bias is dynamic and adjusts based on the number of remaining Tanzanian cities.  
- If no Tanzanian cities remain, the bias is set to 0, and all probabilities are distributed equally among non-Tanzanian cities.  


## 6. **Letter Selection**

```python
def select_best_letter(possible_words, probabilities, feedback, guesses, advanced_rules=False, exhausted_letters=None):
    # ...
```

### **Purpose and Theoretical Background**  
The `select_best_letter` function implements a **decision-making mechanism** that combines **information gain** and **positional value** to select the most informative letter to guess. This function is used in both simple and advanced environments, but its behavior adapts based on the `advanced_rules` flag.  

### **Mathematical Formulation**  
1. **Information Gain**:  
   The information gain for each candidate letter is calculated using the `calculate_information_gain` function:  
   ```python
   info_gains[letter] = calculate_information_gain(possible_words, probabilities, letter, advanced_rules, exhausted_letters)
   ```  

2. **Positional Value**:  
   The positional value measures how well a letter discriminates between cities at specific positions:  
   ```python
   split_ratio = with_letter / total
   position_splits[i] = 4 * split_ratio * (1 - split_ratio)
   ```  

3. **Combined Score**:  
   The final score for each letter is a weighted combination of normalized information gain and positional value:  
   ```python
   norm_gain = info_gains[letter] / max(info_gains.values())
   norm_pos = positional_values[letter] / max(positional_values.values())
   letter_scores[letter] = (info_gain_weight * norm_gain + positional_weight * norm_pos)
   ```  

### **Advanced Rules**  
When `advanced_rules` is enabled, the function considers repeating letters if they are likely to appear multiple times in the word:  
```python
if letter in expected_occurrences and expected_occurrences[letter] > revealed_count + 0.5:
    pass  # Keep the letter as a candidate
```  

### **Exhausted Letters**  
Letters that have been guessed multiple times with no change in feedback are penalized:  
```python
if letter in agent_function.consecutive_repeats and agent_function.consecutive_repeats[letter] > 0:
    letter_scores[letter] *= 0.1  # Apply a 90% penalty
```  

### **Key Insights**  
- The function dynamically adjusts its strategy based on the stage of the game (early vs. late).  
- It prioritizes letters that maximize both information gain and positional discrimination.  
- Advanced rules and exhausted letter handling ensure adaptability to complex scenarios.  


## 7. **Main Agent Function**

```python
def agent_function(request_data, request_info):
    # ...
```

### **Purpose and Theoretical Background**  
The `agent_function` is the **core decision-making engine** of the agent. It integrates all components—filtering, probability updates, and letter selection—to determine the next move. The function operates in both simple and advanced environments, dynamically adapting its behavior based on the feedback and guesses.  

### **State Initialization**  
On the first run, the function initializes state variables, such as the word list and advanced rules flag:  
```python
if not hasattr(agent_function, 'word_list'):
    agent_function.word_list = all_cities
    agent_function.advanced_rules = None
```  

### **Advanced Rules Detection**  
The function detects whether advanced rules are active by analyzing the feedback and guesses:  
```python
if agent_function.advanced_rules is None:
    duplicate_letters = len(letter_guesses) != len(set(letter_guesses))
    letter_mismatch = any(letter in feedback and word.count(letter) > feedback.count(letter) 
                      for word in agent_function.word_list for letter in letter_guesses)
    agent_function.advanced_rules = duplicate_letters or letter_mismatch
```  

### **Feedback Processing**  
The function processes feedback to update the list of possible cities. In the **simple environment**, it uses `filter_words`:  
```python
if not agent_function.advanced_rules:
    possible_words = filter_words(agent_function.word_list, feedback, guesses)
```  
In the **advanced environment**, it uses `filter_words_advanced`:  
```python
else:
    possible_words = filter_words_advanced(agent_function.word_list, feedback, guesses)
```  

### **Probability Updates**  
The probabilities of the remaining cities are updated using the `update_word_probabilities` function:  
```python
probabilities = update_word_probabilities(possible_words)
```  

### **Decision Logic**  
The function employs a **hierarchical decision-making process**:  
1. **Early Word Guessing**:  
   If the number of possible cities is small (≤ 3) or a city has a high probability (> 0.7), the agent guesses the most likely city:  
   ```python
   if len(possible_words) <= 3 or max(probabilities) > 0.7:
       return word_prob_pairs[0][1]  # Guess the most likely city
   ```  

2. **Single Word Remaining**:  
   If only one city remains, the agent guesses it:  
   ```python
   if len(possible_words) == 1:
       return possible_words[0]
   ```  

3. **Letter Selection**:  
   Otherwise, the agent selects the best letter to guess using `select_best_letter`:  
   ```python
   best_letter = select_best_letter(possible_words, probabilities, feedback, guesses, agent_function.advanced_rules, agent_function.exhausted_letters)
   if best_letter:
       return best_letter
   ```  

4. **Fallback Mechanism**:  
   If no candidate letters are available, the agent falls back to guessing common English letters in a predefined order:  
   ```python
   for letter in 'ANIOERULGSHTMKCBDPYQZVJWFX':
       if letter not in guesses:
           return letter
   ```  

### **Key Insights**  
- The function dynamically adapts its strategy based on the feedback and the number of remaining cities.  
- It integrates filtering, probability updates, and letter selection into a cohesive decision-making process.  
- Fallback mechanisms ensure robustness in edge cases.  



## Self Evaluation and Design Decisions

#### **1. Preprocessing on `city_ascii` Column Instead of `cities` Column**
- **Problem**: Initially, preprocessing was applied to the `cities` column, which led to incorrect conversions of city names (e.g., converting "Ä°zmir" to "Azmir" instead of "Izmir").
- **Solution**: The preprocessing was shifted to the `city_ascii` column
- **Impact**: By using `city_ascii`, the city names remain consistent and accurate, avoiding errors in filtering and matching during the game. This is critical for correctly identifying cities based on feedback and guesses.

---

#### **2. Handling Repeated Letter Guesses with `exhausted_letters`**
- **Problem**: In the advanced rules environment, the agent sometimes repeatedly guessed the same letter even when the feedback did not change, leading to wasted guesses.
- **Solution**: The `exhausted_letters` set was introduced to track letters that have been guessed multiple times without any change in feedback. If a letter is in this set, it is excluded from future guesses.
- **Implementation**:
  - The `consecutive_repeats` dictionary tracks how many times a letter has been guessed consecutively without changing the feedback.
  - If a letter is guessed twice with no change in feedback, it is added to `exhausted_letters`.
  - The `select_best_letter` function checks `exhausted_letters` and avoids guessing those letters.
- **Impact**: This significantly reduces wasted guesses, improving the agent's efficiency in the advanced rules environment.

---

#### **3. Handling `-` in Advanced Rules Without Stripping**
- **Problem**: Initially, I attempted to strip `-` characters in the advanced rules environment, which led to incorrect filtering and wrong city guesses.
- **Solution**: The stripping logic was removed, and the `filter_words_advanced` function was updated to handle `-` characters correctly.
- **Implementation**:
  - The `filter_words_advanced` function checks each position in the feedback:
    - If the feedback character is `-`, it allows any character in that position (as long as it doesn't conflict with other constraints).
    - If the feedback character is not `-`, it ensures the word matches the feedback at that position.
  - This approach allows the agent to handle cases where many positions are still hidden (`-`) while still narrowing down the possible words based on revealed letters.
- **Impact**: The agent can now correctly guess city names even when most of the feedback consists of `-` characters.

---

#### **4. Explanation of Weight Values for Information Gain and Positional Value**
- **Weights**:
  - `info_gain_weight = 0.7`
  - `positional_weight = 0.3`
- **Reasoning**:
  - **Information Gain**: Measures how much a letter reduces uncertainty about the possible words. This is the primary metric for selecting letters, as it directly impacts the agent's ability to narrow down the word list.
  - **Positional Value**: Measures how well a letter discriminates between words at specific positions. This is secondary but still important, especially in the late game when fewer positions remain to be revealed.
- **Dynamic Adjustment**:
  - When `revealed_ratio < 0.5` (late game), the weights are adjusted to:
    - `info_gain_weight = 0.6`
    - `positional_weight = 0.4`
  - **Reason**: In the late game, positional discrimination becomes more critical because fewer letters remain to be guessed, and the agent needs to focus on specific positions to identify the correct word.
- **Impact**: This balanced approach ensures the agent prioritizes letters that provide the most information while also considering their positional value, especially in the late game.

---

#### **5. Early Word Guessing to Lower Guess Count**
- **Implementation**:
  - The agent checks two conditions for early word guessing:
    1. If there are 3 or fewer possible words and at least 2 letters have been guessed.
    2. If the highest probability among the possible words exceeds 0.7.
  - If either condition is met, the agent guesses the most probable word from the remaining candidates.
- **Reasoning**:
  - Early word guessing reduces the number of guesses required to win the game, especially when the agent is confident about the correct word.
  - This strategy is particularly effective in the standard rules environment, where the agent can often identify the correct word with high confidence after a few letter guesses.
- **Impact**: This optimization significantly reduces the average number of guesses per game, improving the agent's overall performance.

## Output Format
For Simple environment the code scores: <br >
<img width="523" alt="image" src="https://github.com/user-attachments/assets/16bb8cc8-a728-4984-9534-2d35de691843" />


For Advanced environment the code scores: <br >
<img width="524" alt="image" src="https://github.com/user-attachments/assets/18b2d362-d3c9-49ce-b1ca-efde87098474" />



