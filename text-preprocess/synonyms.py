import requests
import json
import time
import pubchempy as pcp
import pandas as pd

import os
import argparse

parser = argparse.ArgumentParser(description='Retrieve the synonyms of a molecule.')
parser.add_argument('--source', '-s', type=str, default='../data/contrast-pretrain/S/text', help='Source directory containing the molecule files.')
args = parser.parse_args()

# Specify paths
source_dir= args.source
files = os.listdir(source_dir)
    
# Prepare output
dict = pd.DataFrame(columns=["cid", "name", "num_synonyms", "list_synonyms"])

# Load and process text
for file in files:
    if file.endswith('.txt'):

        # Get molecule CID
        cid = file.split('_')[-1].split('.')[0]

        # Request from PubChem
        start_time = time.time()
        mol = pcp.Compound.from_cid(cid)

        # Retrieve synonymcs from the PubChem API
        name = mol.iupac_name
        synonyms = mol.synonyms
        
        # Save names and synonyms
        if synonyms is not None:
            print(f"Molecule {cid}: {name} a.k.a {synonyms[0]} and {len(synonyms)-1} other names.")
            dict = pd.concat([dict, pd.DataFrame({"cid": cid, "name": name, "num_synonyms": len(synonyms), "list_synonyms": str(synonyms)}, index=[0])], ignore_index=True)

        # Wait 0.2 seconds after the start time
        #time.sleep(max(0, start_time + 0.2 - time.time()))

# Save the results to a CSV file
dict.to_csv("synonyms.csv", index=False)