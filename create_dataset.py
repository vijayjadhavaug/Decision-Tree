import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate a dataset with non-linear relationships and balanced classes
num_rows = 1000

# Generate random salary values (in thousands)
salary = np.random.randint(30, 120, num_rows)

# Generate random age values
age = np.random.randint(18, 65, num_rows)

# Generate random education levels (1: High School, 2: Bachelor's, 3: Master's, 4: PhD)
education = np.random.choice([1, 2, 3, 4], num_rows)

# Generate random experience levels (0-40 years)
experience = np.random.randint(0, 40, num_rows)

# Generate random credit score (300-850)
credit_score = np.random.randint(300, 850, num_rows)

# Generate probability scores using a non-linear function
probability_score = (
    salary * 0.02 + age * 0.1 + education * 5 + experience * 0.3 + credit_score * 0.002
)

# Normalize the scores to be between 0 and 1 using min-max scaling
probability_score = (probability_score - probability_score.min()) / (probability_score.max() - probability_score.min())

# Assign purchase labels (0 or 1) based on thresholding at the median to balance the classes
purchase = (probability_score > np.median(probability_score)).astype(int)

# Create a DataFrame
df = pd.DataFrame({
    'Salary': salary,
    'Age': age,
    'Education_Level': education,
    'Experience': experience,
    'Credit_Score': credit_score,
    'Purchase': purchase
})

# Save to CSV
df.to_csv("decision_tree_dataset_balanced.csv", index=False)

# Display the class distribution
print("Class distribution:\n", df['Purchase'].value_counts())

# Display the first few rows
print(df.head())
