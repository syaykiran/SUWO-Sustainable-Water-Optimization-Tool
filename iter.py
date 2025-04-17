import itertools
import pandas as pd

# Define ranges for K1, K2, K3
K1_values = [0.1, 0.5, 1, 5, 10]
K2_values = [0.1, 0.5, 1, 5, 10]
K3_values = [0.1, 0.5, 1, 5, 10]

# Generate all combinations
combinations = list(itertools.product(K1_values, K2_values, K3_values))

# Create a DataFrame
df = pd.DataFrame(combinations, columns=['K1', 'K2', 'K3'])

# Save to Excel
df.to_excel('penalty_combinations.xlsx', index=False)