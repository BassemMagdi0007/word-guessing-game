# word-guessing-game

# Detailed Function Explanations

Below is a detailed explanation of each function in the code, including its purpose, inputs, outputs, and key implementation details. Important notes are highlighted using bullet points.

---

## 1. **Data Preprocessing**

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

- **Purpose:** Prepares the dataset of world cities for use in the guessing game.
- **Details:**
  - The dataset is loaded from a CSV file using `pandas`.
  - City names are converted to uppercase to ensure consistency during comparisons.
  - Cities are separated into Tanzanian and non-Tanzanian lists based on the `iso2` column (country code).
  - A combined list of all cities is created for use in the guessing logic.

**Important Notes:**
- The Tanzanian bias is implemented by prioritizing cities from Tanzania during probability updates.
- The dataset must contain the `city_ascii` and `iso2` columns for this preprocessing to work.

---

## 2. **Information Gain Calculation**

```python
def calculate_information_gain(possible_words, probabilities, letter, advanced_rules=False, exhausted_letters=None):
```

- **Purpose:** Calculates the expected reduction in uncertainty (entropy) when guessing a specific letter.
- **Inputs:**
  - `possible_words`: A list of remaining possible cities.
  - `probabilities`: A list of probabilities associated with each city.
  - `letter`: The letter being evaluated for guessing.
  - `advanced_rules`: A flag to enable advanced guessing rules (e.g., repeated letter guesses).
  - `exhausted_letters`: A set of letters that have been guessed multiple times with no change in feedback.
- **Output:** The information gain for the given letter, measured in bits.

**Implementation Details:**
- The function first checks if there are no possible words, in which case it returns 0.
- If advanced rules are enabled and the letter is in the `exhausted_letters` set, the function returns 0 (no information gain for exhausted letters).
- The current entropy of the system is calculated using the formula:
  ```python
  current_entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
  ```
- For advanced rules, the function groups words based on the possible outcomes of guessing the letter (e.g., positions where the letter appears).
- For standard rules, the function groups words based on patterns of the letter in the word.
- The expected entropy after guessing the letter is calculated by summing the weighted entropies of each outcome group.
- The information gain is the difference between the current entropy and the expected entropy.

**Important Notes:**
- The function handles both standard and advanced rules, making it flexible for different game modes.
- The use of entropy ensures that the agent prioritizes letters that provide the most information.

---

## 3. **Word Filtering**

```python
def filter_words(word_list, feedback, guesses):
```

- **Purpose:** Filters the list of possible cities based on feedback and previous guesses.
- **Inputs:**
  - `word_list`: A list of candidate cities.
  - `feedback`: The current feedback (e.g., revealed letters).
  - `guesses`: A list of previous guesses (both letters and words).
- **Output:** A filtered list of cities that match the feedback and guesses.

**Implementation Details:**
- The function iterates through each word in the `word_list`.
- For each word, it checks if the length matches the feedback length.
- It then verifies if the revealed positions in the feedback match the corresponding letters in the word.
- The function ensures that the word does not contain any guessed letters that are not in the feedback.
- Words that pass all checks are added to the filtered list.

**Important Notes:**
- This function is critical for reducing the search space and improving the agent's efficiency.
- It ensures that only valid candidates are considered for further processing.

---

## 4. **Advanced Word Filtering**

```python
def filter_words_advanced(word_list, feedback, guesses):
```

- **Purpose:** Filters the list of possible cities for advanced rules, allowing for repeated letter guesses and positional feedback.
- **Inputs:**
  - `word_list`: A list of candidate cities.
  - `feedback`: The current feedback (e.g., revealed letters).
  - `guesses`: A list of previous guesses (both letters and words).
- **Output:** A filtered list of cities that match the feedback and guesses under advanced rules.

**Implementation Details:**
- Similar to `filter_words`, but it allows for words of any length up to the feedback length.
- It checks if revealed positions match the corresponding letters in the word.
- The function ensures that the word does not contain any guessed letters that are not in the feedback.
- Words that pass all checks are added to the filtered list.

**Important Notes:**
- This function is used when advanced rules are enabled, allowing for more flexible guessing strategies.
- It supports scenarios where the same letter may appear multiple times in the word.

---

## 5. **Probability Updates**

```python
def update_word_probabilities(word_list):
```

- **Purpose:** Updates the probabilities of each city based on Tanzanian bias.
- **Inputs:**
  - `word_list`: A list of candidate cities.
- **Output:** A list of probabilities corresponding to each city.

**Implementation Details:**
- The function first separates Tanzanian and non-Tanzanian cities from the `word_list`.
- It assigns a higher weight (0.7) to Tanzanian cities and a lower weight (0.3) to non-Tanzanian cities.
- Probabilities are calculated by dividing the weight by the number of cities in each category.
- If no Tanzanian cities are present, the weight is set to 0.

**Important Notes:**
- The Tanzanian bias is dynamic and adjusts based on the remaining candidate cities.
- This function ensures that Tanzanian cities are prioritized during guessing.

---

## 6. **Letter Selection**

```python
def select_best_letter(possible_words, probabilities, feedback, guesses, advanced_rules=False, exhausted_letters=None):
```

- **Purpose:** Selects the best letter to guess based on information gain and positional value.
- **Inputs:**
  - `possible_words`: A list of remaining possible cities.
  - `probabilities`: A list of probabilities associated with each city.
  - `feedback`: The current feedback (e.g., revealed letters).
  - `guesses`: A list of previous guesses (both letters and words).
  - `advanced_rules`: A flag to enable advanced guessing rules.
  - `exhausted_letters`: A set of letters that have been guessed multiple times with no change in feedback.
- **Output:** The best letter to guess.

**Implementation Details:**
- The function first identifies candidate letters that haven't been guessed yet.
- For advanced rules, it considers repeating letters if they are likely to appear multiple times in the word.
- It calculates the information gain and positional value for each candidate letter.
- The final score for each letter is a weighted combination of information gain and positional value.
- The letter with the highest score is selected as the best guess.

**Important Notes:**
- The function uses a combination of information gain and positional value to optimize letter selection.
- It supports both standard and advanced rules, making it versatile for different game modes.

---

## 7. **Main Agent Function**

```python
def agent_function(request_data, request_info):
```

- **Purpose:** The main function that processes feedback and guesses to determine the next move.
- **Inputs:**
  - `request_data`: Contains feedback and guesses.
  - `request_info`: Additional information about the request.
- **Output:** The next guess (either a letter or a city name).

**Implementation Details:**
- The function initializes state variables (e.g., word list, incorrect words) on the first run.
- It detects whether advanced rules are being used based on the feedback and guesses.
- The function filters the list of possible cities based on feedback and guesses.
- It updates the probabilities of each city using `update_word_probabilities`.
- If only one city remains, it guesses that city.
- Otherwise, it selects the best letter to guess using `select_best_letter`.

**Important Notes:**
- The function dynamically adjusts its strategy based on the game mode (standard or advanced rules).
- It ensures that the agent makes optimal guesses by combining filtering, probability updates, and letter selection.

---

## 8. **Execution Flow**

```python
if __name__ == '__main__':
```

- **Purpose:** Runs the agent in a simulated environment.
- **Inputs:** Command-line arguments (e.g., configuration file).
- **Output:** Results of the guessing game.

**Implementation Details:**
- The function sets up logging for debugging and monitoring.
- It runs the agent in a simulated environment using the `client.run` function.
- The agent is executed for a specified number of runs (e.g., 100,000).

**Important Notes:**
- The execution flow is designed for testing and evaluation.
- It allows for parallel runs to improve efficiency during testing.

---

This detailed explanation provides a comprehensive understanding of each function in the code, including its purpose, inputs, outputs, and key implementation details. The agent is designed to be efficient, adaptable, and capable of handling both standard and advanced guessing rules.


# Detailed Function Explanations

Below is a detailed explanation of each function in the code, including their purpose, inputs, outputs, and key design decisions. Important notes are highlighted using bullet points.

---

## 1. **Data Preprocessing**

### Purpose:
The data preprocessing step loads and prepares the city dataset for use in the agent. It ensures consistency in city names and separates Tanzanian cities from non-Tanzanian ones to apply a bias toward Tanzanian cities.

### Inputs:
- A CSV file (`filtered_worldcities.csv`) containing city data.

### Outputs:
- A list of Tanzanian cities (`tanzanian_cities`).
- A list of non-Tanzanian cities (`non_tanzanian_cities`).
- A combined list of all cities (`all_cities`).

### Explanation:
The function reads the city dataset using `pandas` and converts city names to uppercase to ensure consistency during comparisons. It then filters the dataset to separate Tanzanian cities (identified by the ISO2 code 'TZ') from non-Tanzanian cities. These lists are used throughout the agent to prioritize Tanzanian cities when making guesses.

**Important Notes:**
- The Tanzanian bias is dynamic and adjusts based on the remaining possible cities.
- Uppercase conversion ensures case-insensitive matching during filtering and guessing.

---

## 2. **Information Gain Calculation**

### Purpose:
This function calculates the expected reduction in uncertainty (entropy) when guessing a specific letter. It helps the agent choose the letter that provides the most information about the remaining possible cities.

### Inputs:
- `possible_words`: A list of remaining possible cities.
- `probabilities`: A list of probabilities associated with each city.
- `letter`: The letter being evaluated.
- `advanced_rules`: A flag indicating whether advanced rules are active.
- `exhausted_letters`: A set of letters that have been guessed multiple times with no change in feedback.

### Outputs:
- The information gain for the given letter (a float value).

### Explanation:
The function calculates the current entropy of the system based on the probabilities of the remaining cities. It then simulates the expected entropy after guessing the letter by considering all possible outcomes (e.g., the letter appearing in specific positions or not appearing at all). The information gain is the difference between the current entropy and the expected entropy.

**Important Notes:**
- Advanced rules allow the agent to handle scenarios where letters may appear multiple times in a city name.
- Exhausted letters (those guessed multiple times with no change in feedback) are ignored to avoid redundant guesses.

---

## 3. **Word Filtering**

### Purpose:
This function filters the list of possible cities based on the current feedback and previous guesses. It ensures that only cities consistent with the feedback and guesses are considered.

### Inputs:
- `word_list`: A list of candidate cities.
- `feedback`: The current feedback (e.g., revealed letters).
- `guesses`: A list of previous guesses.

### Outputs:
- A filtered list of cities that match the feedback and guesses.

### Explanation:
The function iterates through the list of candidate cities and checks each city against the feedback and guesses. A city is retained if:
1. Its length matches the feedback length.
2. All revealed letters in the feedback match the corresponding positions in the city name.
3. It does not contain any guessed letters that are not in the feedback.

**Important Notes:**
- The function ensures that the agent only considers cities that are logically possible given the feedback.
- It handles edge cases, such as when a guessed letter appears in the feedback but not at the expected position.

---

## 4. **Advanced Word Filtering**

### Purpose:
This function extends the basic word filtering to handle advanced rules, such as repeated letter guesses and positional feedback. It allows the agent to consider cities of varying lengths and repeated letter occurrences.

### Inputs:
- `word_list`: A list of candidate cities.
- `feedback`: The current feedback (e.g., revealed letters).
- `guesses`: A list of previous guesses.

### Outputs:
- A filtered list of cities that match the feedback and guesses under advanced rules.

### Explanation:
The function is similar to the basic word filtering function but relaxes some constraints to accommodate advanced rules. For example:
- Cities shorter than the feedback length are considered if they match the revealed letters.
- Repeated letter occurrences are handled by checking if the city contains at least as many occurrences of a letter as revealed in the feedback.

**Important Notes:**
- This function is crucial for handling scenarios where the same letter may appear multiple times in a city name.
- It ensures the agent can adapt to more complex feedback patterns.

---

## 5. **Probability Updates**

### Purpose:
This function updates the probabilities of each city based on the Tanzanian bias and the remaining possible cities. It ensures that Tanzanian cities are prioritized when making guesses.

### Inputs:
- `word_list`: A list of candidate cities.

### Outputs:
- A list of probabilities corresponding to each city.

### Explanation:
The function calculates the probability of each city by considering whether it is Tanzanian or non-Tanzanian. Tanzanian cities are assigned a higher weight (0.7 by default), while non-Tanzanian cities share the remaining probability mass. The probabilities are normalized to sum to 1.

**Important Notes:**
- The Tanzanian bias is dynamic and adjusts based on the number of remaining Tanzanian cities.
- If no Tanzanian cities remain, the bias is set to 0, and all probabilities are distributed equally among non-Tanzanian cities.

---

## 6. **Letter Selection**

### Purpose:
This function selects the best letter to guess based on a combination of information gain and positional value. It ensures that the agent chooses the most informative letter while considering the likelihood of the letter appearing in specific positions.

### Inputs:
- `possible_words`: A list of remaining possible cities.
- `probabilities`: A list of probabilities associated with each city.
- `feedback`: The current feedback (e.g., revealed letters).
- `guesses`: A list of previous guesses.
- `advanced_rules`: A flag indicating whether advanced rules are active.
- `exhausted_letters`: A set of letters that have been guessed multiple times with no change in feedback.

### Outputs:
- The best letter to guess (a single character).

### Explanation:
The function evaluates each candidate letter by calculating its information gain and positional value. The information gain measures how much the letter reduces uncertainty, while the positional value measures how well the letter discriminates between cities at specific positions. The final score for each letter is a weighted combination of these two metrics.

**Important Notes:**
- The function prioritizes letters that are likely to appear in multiple positions.
- Exhausted letters are penalized to avoid redundant guesses.
- The weights for information gain and positional value are adjusted based on the stage of the game (early vs. late).

---

## 7. **Main Agent Function**

### Purpose:
This is the core function that processes feedback and guesses to determine the agent's next move. It integrates all the components of the agent, including filtering, probability updates, and letter selection.

### Inputs:
- `request_data`: Contains feedback and guesses.
- `request_info`: Additional information about the request.

### Outputs:
- The next guess (either a letter or a city name).

### Explanation:
The function initializes state variables (e.g., word list, incorrect words) on the first run. It then filters the list of possible cities based on the feedback and guesses, updates the probabilities, and selects the best letter or city to guess. The function also handles advanced rules, such as repeated letter guesses and positional feedback.

**Important Notes:**
- The function dynamically adjusts its strategy based on the feedback and the number of remaining cities.
- It includes fallback mechanisms to handle edge cases, such as when no candidate letters are available.

---

## 8. **Execution Flow**

### Purpose:
This block runs the agent in a simulated environment. It sets up logging and integrates with the provided `client.py` to handle multiple runs.

### Inputs:
- Command-line arguments (e.g., configuration file).

### Outputs:
- Results of the guessing game.

### Explanation:
The execution flow initializes logging and runs the agent using the `client.py` script. It supports parallel runs for performance evaluation and includes a run limit to control the number of simulations.

**Important Notes:**
- Logging is configured to provide detailed output for debugging and analysis.
- The run limit can be adjusted for testing or performance evaluation.

---

This detailed explanation provides a comprehensive understanding of each function in the code, including their purpose, inputs, outputs, and key design decisions. The agent is designed to be efficient, adaptable, and capable of handling both standard and advanced guessing rules.
