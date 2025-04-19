#Download PaDEL-Descriptor
import pandas as pd
from padelpy import padeldescriptor

#Load bioactivity data from part 1 and part 2

bioactivity_data = "https://raw.githubusercontent.com/RebeccaRKS1998/Bio_Project/refs/heads/main/acetylcholinesterase_04_bioactivity_data_3class_pIC50.csv"

df3 = pd.read_csv(bioactivity_data)
print(df3)

#Selecting canonical smiles and molecule chembl ID columns
selection = ['canonical_smiles', 'molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

print(df3_selection)


#Calculate fingerprint descriptors
padeldescriptor(
    mol_dir='molecule.smi',
    d_file='descriptors_output.csv',
    fingerprints=True,
    retain3d=False,
    retainorder=True,
    threads=2
)

#Prepare X (fingerprint descriptors) and Y Matrix

df3_X = pd.read_csv('descriptors_output.csv')
df3_X = df3_X.drop(columns=['Name'])
print(df3_X)


#Y variable 
df3_Y = df3['pIC50']
print(df3_Y)

#Combine X and Y 
dataset3 = pd.concat([df3_X, df3_Y], axis=1)
print(dataset3)

dataset3.to_csv('acetylecholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', index=False)