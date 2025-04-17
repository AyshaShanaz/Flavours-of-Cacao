# Data Cleaning and Consistency Checks for the Chocolate Bar Ratings Dataset

import pandas as pd
import numpy as np

# Load the dataset from the Excel file
chocolate_df = pd.read_excel('flavors_of_cacao.xlsx')

# Print the first few rows and summary info for an overview
print('Initial Data Head:')
print(chocolate_df.head())

print('\nDataFrame Info:')
print(chocolate_df.info())

# Check for missing values across columns
print('\nMissing Values Count:')
missing_counts = chocolate_df.isnull().sum()
print(missing_counts)

# First, let's fix the header issue by setting the first row as the header
chocolate_df.columns = chocolate_df.iloc[0]
chocolate_df = chocolate_df.drop(0)
chocolate_df = chocolate_df.reset_index(drop=True)

print('\nFixed Header - Data Head:')
print(chocolate_df.head())

# Consistency checks for rating values
# Assuming ratings should be in a 1-5 scale
if 'Rating' in chocolate_df.columns:
    chocolate_df['Rating'] = pd.to_numeric(chocolate_df['Rating'], errors='coerce')
    invalid_ratings = chocolate_df[(chocolate_df['Rating'] < 1) | (chocolate_df['Rating'] > 5)]
    print('\nEntries with invalid Rating values:')
    print(invalid_ratings.shape[0])
    if invalid_ratings.shape[0] > 0:
        print(invalid_ratings[['Rating']].head())
else:
    print('Rating column not found in the dataset.')

# Consistency checks for cocoa percent
# Remove the '%' sign if present and convert to numeric
if 'Cocoa_Percent' in chocolate_df.columns:
    def clean_cocoa(cocoa_str):
        if isinstance(cocoa_str, str):
            return float(cocoa_str.strip('%'))
        return cocoa_str
    chocolate_df['Cocoa_Percent'] = chocolate_df['Cocoa_Percent'].apply(clean_cocoa)
    chocolate_df['Cocoa_Percent'] = pd.to_numeric(chocolate_df['Cocoa_Percent'], errors='coerce')
    # Check range for cocoa percentage; normally it should be between 0 and 100
    invalid_cocoa = chocolate_df[(chocolate_df['Cocoa_Percent'] < 0) | (chocolate_df['Cocoa_Percent'] > 100)]
    print('\nEntries with invalid Cocoa Percent values:')
    print(invalid_cocoa.shape[0])
    if invalid_cocoa.shape[0] > 0:
        print(invalid_cocoa[['Cocoa_Percent']].head())
else:
    print('Cocoa_Percent column not found in the dataset.')

# Check for duplicate entries based on a unique identifier if available e.g., REF
if 'REF' in chocolate_df.columns:
    duplicates = chocolate_df[chocolate_df.duplicated(subset='REF', keep=False)]
    print('\nNumber of duplicate entries based on REF:')
    print(duplicates.shape[0])
else:
    print('REF column not found to check duplicates.')

# Trim string columns to remove leading/trailing spaces for consistency
str_cols = chocolate_df.select_dtypes(include=['object']).columns
for col in str_cols:
    chocolate_df[col] = chocolate_df[col].str.strip()

# Display summary statistics for numerical columns for further inspection
print('\nSummary Statistics:')
print(chocolate_df.describe(include='all'))

# Save the cleaned dataset
chocolate_df.to_csv('cleaned_chocolate_data.csv', index=False)
print('\nCleaned data saved to cleaned_chocolate_data.csv')

print('Data cleaning and consistency checks complete.')
