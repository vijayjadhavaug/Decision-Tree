import numpy as np
import pandas as pd

# Generate a new dataset with 500 points in a U-shape pattern
np.random.seed(42)
X1 = np.random.randn(250, 2) * [2, 1] + [-3, 2]  # First cup shape
X2 = np.random.randn(250, 2) * [2, 1] + [3, -2]  # Second cup shape

# Combine into a DataFrame
X = np.vstack((X1, X2))
y = np.array([0] * 250 + [1] * 250)  # Labels: 0 for first cluster, 1 for second cluster

df_500 = pd.DataFrame({"Feature_1": X[:, 0], "Feature_2": X[:, 1], "Target": y})

# Save the dataset to CSV
df_500.to_csv("custom_u_shape_500.csv", index=False)
print("Dataset saved as custom_u_shape_500.csv")
