from sklearn.datasets import load_iris

df = load_iris(as_frame=True).frame

new_df = df[df["target"] == 0]

print(f'Shape: {new_df.shape}')
print("\nFirst 5 rows:")
print(new_df.head())

print("\nClass name for target 0:", load_iris().target_names[0])