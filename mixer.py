import csv
import glob
import random

# Step 1: Find all *_train_with_lang.tsv files
input_files = glob.glob("*_train_with_lang.tsv")
combined_rows = []

# Step 2: Read and combine all rows
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Skip header
        for row in reader:
            combined_rows.append(row)

# Step 3: Shuffle the combined dataset
random.shuffle(combined_rows)

# Step 4: Write to a new combined file
output_file = "all_languages_train_shuffled.tsv"

with open(output_file, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    # Write the header once
    writer.writerow(["text", "label", "language"])
    writer.writerows(combined_rows)

print(f"Combined and shuffled file saved as: {output_file}")
