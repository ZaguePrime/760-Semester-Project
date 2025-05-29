import requests
import csv

language_id = 'ibo'

# Step 1: Download the file
url = f"https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/{language_id}/train.tsv"
response = requests.get(url)

if response.status_code == 200:
    original_filename = f"{language_id}_train.tsv"
    modified_filename = f"{language_id}_train_with_lang.tsv"

    # Save original
    with open(original_filename, "wb") as f:
        f.write(response.content)
    print("Download completed successfully.")

    # Step 2: Read original TSV and add a new column
    with open(original_filename, "r", encoding="utf-8") as infile, \
         open(modified_filename, "w", encoding="utf-8", newline='') as outfile:

        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        header = next(reader)
        header.append("language")
        writer.writerow(header)

        for row in reader:
            row.append(language_id)
            writer.writerow(row)

    print(f"Modified file written to {modified_filename}")

else:
    print(f"Failed to download file. Status code: {response.status_code}")
