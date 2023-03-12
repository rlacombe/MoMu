import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--directory', '-d', type=str, default='../data/XL/text', help='The directory containing the text files')
args = parser.parse_args()


# Specify path, initialize files and store results
dir_path= args.directory
files = os.listdir(dir_path)
num_files = len(files)
data = pd.DataFrame(columns=["cid", "size", "paragraphs"])

# Loop through each text file in the directory
for i, filename in enumerate(os.listdir(dir_path)):
    if filename.endswith(".txt"):

        # Get molecule CID
        cid = int(filename.split("_")[1].split(".")[0])

        # Get the size of the file in bytes
        size = os.path.getsize(os.path.join(dir_path, filename))

        # Get the number of lines in the file
        lines = sum(1 for line in open(os.path.join(dir_path, filename)))
        
        # Add the molecule cid, file size, and number of lines 
        print(f"File {i+1} out of {num_files}. CID: {cid}, size: {size}, paragraphs: {lines}.")
        data = pd.concat([data, pd.DataFrame({"cid": cid, "size": size, "paragraphs": lines}, index=[0])], ignore_index=True)

# Save the results to a CSV file
data.to_csv("file_stats.csv", index=False)
