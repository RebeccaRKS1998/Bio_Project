#Install the ChEMBL web service package

#Import necessary libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client


#Search for target protein 

#Target search for coronavirus
target = new_client.target 
target_query = target.search('acetylcholinesterase')
#Convert to a DataFrame
targets = pd.DataFrame.from_dict(target_query)
#Display the DataFrame 
print(targets)

#Select and retrieve bioactivity data for Human Acetylcholinesterase 
selected_target = targets.target_chembl_id[1]
print(selected_target)

#Retrieve bioactivity data for CHEMBL220(selected_target)
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df = pd.DataFrame.from_dict(res)
print(df)
#Save bioactivity data to CSV file: bioactivity_data.csv
df.to_csv('acetylcholinesterase_01_bioactivity_data_raw.csv', index=False)


#Deleting missing data for the standard_value and canonical_smiles 
df2 = df[df.standard_value.notna()]
df2 = df2[df.canonical_smiles.notna()]
print(df2)
#Removing duplicates
df2_nr = df2.drop_duplicates(['canonical_smiles'])
print(df2_nr)
#Data pre-processing of bioactivity data by combining: molecule_chembl_id,canonical_smiles,standard_value and bioactivity_class into a DataFrame
selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
df3 = df2_nr[selection]
print(df3)

#Save bioactivity data to CSV file: bioactivity_data_preprocessed.csv
df3.to_csv('acetylcholinesterase_02_bioactivity_data_preprocessed.csv', index=False)


#Labeling compounds as active (<1000nM), inactive (10,000nM), or intermediate (1,000-10,000nM) based on IC50 data
df4 = pd.read_csv('acetylcholinesterase_02_bioactivity_data_preprocessed.csv')

bioactivity_threshold = []
for i in df4.standard_value:
    if float(i) >= 10000:
        bioactivity_threshold.append("inactive")
    elif float(i) <= 1000:
        bioactivity_threshold.append("active")
    else:
        bioactivity_threshold.append("intermediate")

bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df5 = pd.concat([df4, bioactivity_class], axis=1)
print(df5)

#Save bioactivity data to CSV file: bioactivity_data_preprocessed.csv
df5.to_csv('acetylcholinesterase_03_bioactivity_data_curated.csv', index=False)
 






