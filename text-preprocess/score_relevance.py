import nltk
nltk.download('punkt')
import os
from nltk.tokenize import sent_tokenize
import numpy as np

import requests
import json

import time

# Find the name and synomyms of molecules
def get_molecule_info(cid): # Find the name and synomyms of molecules
    """
    Retrieves the synonyms of a molecule from the PubChem API.
  
    Args:
    - cid (int): the Compound ID of the molecule on PubChem (i.e. file number)

    Returns:
    - synonyms (str list): a list of names used for the molecule

    (!) PubChem API limits requests to 5/sec.
    """
    
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/synonyms/json".format(cid)
    response = requests.get(base_url)
    if response.status_code == 200:
        data = json.loads(response.content.decode('utf-8'))
        synonyms = data['InformationList']['Information'][0]['Synonym']
        return synonyms
    else:
        print('Error retrieving molecule info: {}'.format(response.text))
        return None
    
# Choose relevance scoring method
def score_relevance(sentence, molecule, method):
    """Determines the relevance score of a given sentence based on the chosen scoring method.

    Args:
    - sentence (str): The sentence to be scored.
    - molecule (str): The molecule to be searched for in the sentence.
    - method (str): The chosen relevance scoring method. It can either be "dummy" or "counting".

    Returns:
    - An integer representing the relevance score of the sentence. 
        Method is "dummy": score is always 0. 
        Method is "counting": the score is the number of times a synonym for the molecule appears in the sentence.
    """
    relevance_funcs = {
        'dummy': lambda: 0,
        'counting': lambda: count_molecule_mentions(sentence, molecule)
    }
    return relevance_funcs.get(method, lambda: 0)()

def parser(method, text_name): # Text file name e.g. text_1.txt
    """
    Parses the file <text_name> to extract only the most relevant sentences.
  
    Args:
    - text_name (str): filename

    Returns:    
    - sent_list (List[List[str]]): the list of paragraphs (list of list of sentences).
    - sent_scores (List[List[int/float]]): the score for each sentence in the list of list of sentences.
    """
    # Load and process text
    text_path = os.path.join('MoMu/Pretrain/data/text/', text_name)

    cid = text_name.split('_')[-1].split('.')[0]

    sent_list = []
    sent_scores = []
    count = 0
  
    molecule_names = get_molecule_info(cid) # Replace by dictionary once built
    time.sleep(0.2) # To avoid overwhelming API - replace by dictionary
  
    # Parse each line/paragraph in the text file
    for l, line in enumerate(open(text_path, 'r', encoding='utf-8')):
    #  if l > 500:
    #    break
  
    sentences = sent_tokenize(line)
    sent_list.append(sentences)
    sent_scores.append([0] * len(sentences))

    for s, sent in enumerate(sentences):
        sent_scores[l][s] = score_relevance(sent, molecule_names, method)     

    return sent_list, sent_scores

def extract_relevant_sentences(method, text_filename):
    """
    Extracts relevant sentences from a given text file using the chosen method.

    Args:
    - method (str): The chosen relevance scoring method. It can either be "dummy" or "counting".
    - text_filename (str): The path to the text file to be processed.

    Returns:
    - A tuple containing the following four elements:
      1. relevant_sentences (List[List[str]]): A list of lists of relevant sentences. Each inner list contains 
         the relevant sentences for a single paragraph in the text file.
      2. num_relevant (int): The total number of relevant sentences extracted from the text file.
      3. total_sentences (int): The total number of sentences in the text file.
      4. num_paragraphs (int): The number of paragraphs in the text file.
    """

    sent_lists, sent_scores = parser(method, text_filename)
    print(f"Text file: {text_filename}")
    print(f"Method: {method}")
    lengths = np.array([len(sent_list) for sent_list in sent_lists])
    total_sentences = np.sum(lengths)
    print(f"Found {len(sent_lists)} paragraphs with an average of {np.mean(lengths)} sentences per paragraph, for a total of {total_sentences} sentences.")

    # Return only the directly relevant sentences
    relevant_sentences = sent_lists
    for i, sent_list in enumerate(sent_lists):
        count = 0
        for j, is_relevant in enumerate(sent_scores[i]):
        if is_relevant == 0:
            relevant_sentences[i].pop(j - count)
            count += 1
  
      num_relevant = sum( [ len(line) for line in relevant_sentences])

    print(f"Extracted a total of {num_relevant} relevant sentences.")
    percent_kept = '{:.1f}%'.format((num_relevant / total_sentences)*100)
    print(f"Keeping {percent_kept} of total sentences.")
    return relevant_sentences, num_relevant, total_sentences, len(sent_lists)

def count_molecule_mentions(sentence, molecule_names):
    """
    Counts the number of times any of the synonyms for one molecule appears in a given sentence, and returns the 
    number of counts.

    Args:
    - sentence (str): The sentence to be searched for molecule mentions.
    - molecule_names (List[str]): A list of synonyms for a single molecule.

    Returns:
    - count (int): the number of times any of the synonyms for the molecule appears in the sentence.
    """
    count = 0
    sentence = sentence.lower()  # Convert everything to lowercase
    for name in molecule_names:
        count += sentence.count(name.lower())
    return count

