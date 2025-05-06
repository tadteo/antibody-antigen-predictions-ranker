import pandas as pd

# Load the manifest
df = pd.read_csv('data/manifest_with_ptm.csv')

# Group by 'bucket' and calculate average weight
summary = df.groupby('bucket').agg(
    entry_count=('bucket', 'size'),
    total_weight=('weight', 'sum'),
    avg_weight=('weight', 'mean')
)

print(summary)
