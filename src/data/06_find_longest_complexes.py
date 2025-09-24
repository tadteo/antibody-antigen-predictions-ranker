#!/usr/bin/env python3
"""
This script is designed to find the longest complexes in a CSV file and write them to a new CSV file.

to run:
python find_longest_complexes.py --input-csv input_path_with_original_csv --output-csv output_path_with_longest_complexes_csv
"""

import csv
import os # Added for path manipulation

# Path to the input CSV file
input_csv_file_path = "/proj/berzelius-2021-29/users/x_matta/antibody-antigen-predictions-ranker/src/data/data/manifest_new_with_filtered_test_density_with_clipping.csv"
# Determine the output file path in the same directory as the input
base_name_without_ext, ext = os.path.splitext(os.path.basename(input_csv_file_path))
output_filename = f"{base_name_without_ext}_500k_maxlen{ext}"
output_csv_file_path = os.path.join(os.path.dirname(input_csv_file_path), output_filename)

complex_max_lengths = {}
MAX_ALLOWED_LEN_SAMPLE = 500000

print(f"Reading input CSV: {input_csv_file_path}")
try:
    with open(input_csv_file_path, 'r', newline='') as file: # Added newline='' for csv handling
        reader = csv.DictReader(file)
        if not reader.fieldnames or "complex_id" not in reader.fieldnames or "len_sample" not in reader.fieldnames:
            print(f"Error: CSV file must contain 'complex_id' and 'len_sample' columns.")
            exit()
            
        for row_number, row in enumerate(reader, 1):
            try:
                complex_id = row["complex_id"]
                if not complex_id:
                    # print(f"Skipping row {row_number} due to empty complex_id.") # Optional: less verbose
                    continue

                len_sample_str = row["len_sample"]
                if not len_sample_str:
                    # print(f"Skipping row {row_number} (complex_id: {complex_id}) due to empty len_sample.") # Optional: less verbose
                    continue
                
                len_sample = int(len_sample_str)

                if complex_id in complex_max_lengths:
                    if len_sample > complex_max_lengths[complex_id]:
                        complex_max_lengths[complex_id] = len_sample
                else:
                    complex_max_lengths[complex_id] = len_sample
            except ValueError:
                # print(f"Skipping row {row_number} (complex_id: {row.get('complex_id', 'N/A')}) due to invalid non-integer len_sample: '{row.get('len_sample', 'N/A')}'") # Optional: less verbose
                continue
            except KeyError as e:
                print(f"Skipping row {row_number} due to missing key: {e}. Row content: {row}")
                continue

except FileNotFoundError:
    print(f"Error: The file {input_csv_file_path} was not found.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the file for max lengths: {e}")
    exit()

if not complex_max_lengths:
    print("No data processed to determine max lengths. The complex_max_lengths dictionary is empty.")
    exit()

# Identify complexes to keep (max len_sample <= MAX_ALLOWED_LEN_SAMPLE)
complexes_to_keep = set()
for complex_id, max_len in complex_max_lengths.items():
    if max_len <= MAX_ALLOWED_LEN_SAMPLE:
        complexes_to_keep.add(complex_id)

print(f"Identified {len(complexes_to_keep)} complexes with max len_sample <= {MAX_ALLOWED_LEN_SAMPLE}.")
if not complexes_to_keep:
    print(f"No complexes found with max len_sample <= {MAX_ALLOWED_LEN_SAMPLE}. Output file will be empty or not created.")
    # Decide if an empty file should be created or not
    # For now, let's create an empty file with headers if no complexes meet the criteria
    # To prevent this, one might add an exit() here or handle it before opening the output file.

# Second pass: Read input CSV again and write filtered rows to the new CSV
print(f"Writing filtered data to: {output_csv_file_path}")
try:
    with open(input_csv_file_path, 'r', newline='') as infile, \
         open(output_csv_file_path, 'w', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        if not reader.fieldnames: # Should have been caught earlier, but good for safety
            print("Error: Input CSV fieldnames are missing for the second pass.")
            exit()
            
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        rows_written = 0
        for row in reader:
            if row.get("complex_id") in complexes_to_keep:
                writer.writerow(row)
                rows_written += 1
        print(f"Successfully wrote {rows_written} rows to {output_csv_file_path}.")

except FileNotFoundError: # Should not happen if first pass was successful
    print(f"Error: The file {input_csv_file_path} was not found during the second pass.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while writing the output file: {e}")
    exit()

print("Script finished.")
