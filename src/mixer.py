import csv
import glob
import random

# configuration
input_files = glob.glob("../language_datasets/*_train_with_lang.tsv")
combined_rows = []

# read and combine all input files
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            combined_rows.append(row)

random.shuffle(combined_rows)

output_file = "../language_datasets/all_languages_train_shuffled.tsv"

with open(output_file, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["text", "label", "language"])
    writer.writerows(combined_rows)

print(f"Combined and shuffled file saved as: {output_file}")
