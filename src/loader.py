import os
import requests
import csv

# configuration
language_id = 'ibo'
save_dir = 'language_datasets'  

os.makedirs(save_dir, exist_ok=True)

original_filename = os.path.join(save_dir, f"{language_id}_train.tsv")
modified_filename = os.path.join(save_dir, f"{language_id}_train_with_lang.tsv")

url = f"https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data/{language_id}/train.tsv"
response = requests.get(url)

# augment data
if response.status_code == 200:
    with open(original_filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to {original_filename}")

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
