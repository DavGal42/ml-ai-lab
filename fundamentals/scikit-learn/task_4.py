"""
- Filter the DataFrame to keep only rows where the target is 0
- Load the Iris dataset as a Pandas DataFrame (as_frame=True)
- Print the shape of the filtered DataFrame and the first 5 rows
- Print the class name corresponding to target 0
"""

from sklearn.datasets import load_iris

df = load_iris(as_frame=True).frame

new_df = df[df["target"] == 0]

print(f'Shape: {new_df.shape}')
print("\nFirst 5 rows:")
print(new_df.head())

print("\nClass name for target 0:", load_iris().target_names[0])