import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--directory', '-d', type=str, default='../Pretrain/data/text', help='The directory containing the text files')
parser.add_argument('--method', '-m', type=str, default='counting', help='The method for scoring sentence relevance')
parser.add_argument('--num', '-n', type=int, default=89, help='How many files to parse')
args = parser.parse_args()

directory = args.directory
method = args.method
num_files = args.num

files = sorted(os.listdir(directory))
dir_files = len(files)

print(f"START: parsing the first {min(num_files,dir_files)} out of {n} total files.\n")

total_relevant = 0
total_sentences = 0
total_paragraphs = 0
total_relevant_paragraphs = 0

for i, text_filename in enumerate(files):
    # Executing for only the first 5 files
    if i >= x:
        break
    relevant_sentences, num_rel_sentences, num_sentences, num_paragraphs = extract_relevant_sentences(method, text_filename)
    print(f"--------- Done with {i+1} out of {n} files --------- \n")

    total_relevant += num_rel_sentences
    total_sentences +=  num_sentences
    total_paragraphs += num_paragraphs
    total_relevant_paragraphs += len(relevant_sentences)

print(f"--------- PROCESSED {min(x,n)} FILES ---------")
print(f"Extracted a total of {total_relevant} out of {total_sentences} sentences.")
print(f"Extracted a total of {total_relevant_paragraphs} out of {total_paragraphs} paragraphs.")
percent_kept = '{:.1f}%'.format((total_relevant / total_sentences)*100)
print(f"Keeping {percent_kept} of total sentences.")
