import os
import argparse

parser = argparse.ArgumentParser(description='Clip dataset to max 500 paragraphs per molecule.')
parser.add_argument('--source', '-s', type=str, default='../data/XL/text', help='The source directory containing the original files')
parser.add_argument('--target', '-t', type=str, default='../data/XL-clip/text', help='The target directory containing the clipped files')
args = parser.parse_args()

# Specify paths
source_dir= args.source
target_dir= args.target

files = os.listdir(source_dir)

# Loop through all the files in the source directory
for i, filename in enumerate(files):
    if filename.endswith('.txt'):

        # Open the source file
        with open(os.path.join(source_dir, filename), 'r') as source_file:

            # Read the first 500 lines of the file
            first_500_lines = source_file.readlines()[:500]

        # Create the target file
        with open(os.path.join(target_dir, filename), 'w') as target_file:
        
            # Write the first 500 lines to the target file
            target_file.writelines(first_500_lines)

        print(f"File {i+1} out of {len(files)}.")