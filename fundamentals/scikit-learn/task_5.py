from sklearn.datasets import load_iris

df = load_iris(as_frame=True).frame

counts = df["target"].value_counts().sort_index()

print("Samples per target:")
print(counts)

print("\nReadable class distribution:")
for target_value, count in counts.items():
    class_name = load_iris().target_names[target_value]
    print(f'{class_name}: {count}')