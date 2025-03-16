import pandas as pd
from unidecode import unidecode

# Load the CSV file
cities_df = pd.read_csv('worldcities.csv')

# Filter cities with population >= 100,000
cities_df = cities_df[cities_df['population'] >= 100000]

# Filter cities with no diacritics, hyphens, or spaces in their original name
# Check if the original city name is identical to its unidecode version and has no spaces or hyphens
cities_df = cities_df[cities_df['city_ascii'].apply(lambda x: x == unidecode(x) and ' ' not in x and '-' not in x)]

# Further check that the name only contains A-Z letters
cities_df = cities_df[cities_df['city_ascii'].str.match(r'^[A-Za-z]+$')]

# Save the filtered data to a new CSV file
cities_df.to_csv('filtered_worldcities.csv', index=False)

print("Filtered CSV file saved as 'filtered_worldcities.csv'")