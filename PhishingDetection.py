import pandas as pd

# clean data
clean_lines = []
with open('urlset.csv', 'rb') as file:  # Open in binary mode
    for line in file:
        try:
            clean_lines.append(line.decode('utf-8'))  # Try to decode each line
        except UnicodeDecodeError:
            pass  # Skip lines that cause decoding errors

with open('cleaned_urlset.csv', 'w', encoding='utf-8') as clean_file:
    clean_file.writelines(clean_lines)

# Step 1: Load the dataset
df = pd.read_csv('cleaned_urlset.csv')  

# Remove unnecessary columns
df = df[['domain', 'label']]

