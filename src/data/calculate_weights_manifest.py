import pandas as pd
import sys

def calculate_weights_sum(csv_file):
    # Read the CSV, pandas will pick up the first line as column names
    df = pd.read_csv(csv_file)
    
    # Make sure the necessary columns are present
    if 'complex_id' not in df.columns or 'weight' not in df.columns:
        print("Error: CSV must contain 'complex_id' and 'weight' columns")
        sys.exit(1)
    
    # Group by complex_id and sum the weight column
    weight_sums = df.groupby('complex_id')['weight'].sum()
    
    # Print the results in "complex_id: weight" format
    print("Weights sum per complex ID:")
    print("===========================")
    for complex_id, total_weight in weight_sums.items():
        print(f"{complex_id}: {total_weight}")

    # Print the total weight
    print(f"Total weight: {weight_sums.sum()}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    calculate_weights_sum(csv_file)
