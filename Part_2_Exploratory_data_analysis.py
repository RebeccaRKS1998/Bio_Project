#Install conda and rdkit

#Load bioactivity data
import pandas as pd

bioactivity_data = "https://raw.githubusercontent.com/RebeccaRKS1998/Bio_Project/refs/heads/main/acetylcholinesterase_03_bioactivity_data_curated.csv"

df = pd.read_csv(bioactivity_data)
print(df)

#Orientate Canonical smiles to last column 
df_no_smiles = df.drop(columns = 'canonical_smiles')
smiles = []

for i in df.canonical_smiles.tolist():
    cpd = str(i).split('.')
    cpd_longest = max(cpd, key = len)
    smiles.append(cpd_longest)

smiles = pd.Series(smiles, name = 'canonical_smiles')

df_clean_smiles = pd.concat([df_no_smiles, smiles], axis=1)
print(df_clean_smiles)


#Lipinski descriptors (ADME) (Molecular weight <500 Da, Octanol-water partition coefficient LogP <5, Hydrogen bond donors <5, Hydrogen bond acceptors <10)
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski 

#Calculate descriptors. Inspired by https://codeocean.com/explore/capsules?query=tag:data-curation

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors


df_lipinski = lipinski(df_clean_smiles.canonical_smiles)
print(df_lipinski) 
 
#Combine DataFrames df_lipinski and df
df_combined = pd.concat([df,df_lipinski], axis=1)
print(df_combined)








#Convert IC50 to Logarithmic scale (pIC50) https://github.com/chaninlab/estrogen-receptor-alpha-qsar/blob/master/02_ER_alpha_RO5.ipynb
import numpy as np


# Function to normalize standard_value
def norm_value(input_df):
    input_df = input_df.copy()  # Avoid modifying the original DataFrame
    input_df['standard_value_norm'] = input_df['standard_value'].apply(lambda x: min(x, 100000000))
    input_df = input_df.drop(columns=['standard_value'])
    return input_df

# Function to compute pIC50
def pIC50(input_df):
    input_df = input_df.copy()  # Avoid modifying original DataFrame
    input_df['pIC50'] = input_df['standard_value_norm'].apply(lambda x: -np.log10(x * 10**-9))
    
    # Drop 'standard_value_norm' correctly
    x = input_df.drop('standard_value_norm', axis=1)
    
    return x

# Assuming df_combined exists and has 'standard_value' column
print(df_combined.standard_value.describe())

df_norm = norm_value(df_combined)
print(df_norm)

print(df_norm.standard_value_norm.describe())

df_final = pIC50(df_norm)
print(df_final)

print(df_final.pIC50.describe())


#Convert to CSV 
df_final.to_csv('acetylcholinesterase_04_bioactivity_data_3class_pIC50.csv')



#Removing the intermediate bioactivity class
df_2class = df_final[df_final['class'] != 'intermediate']
print(df_2class)

#Convert to CSV
df_2class.to_csv('acetylcholinesterase_05_bioactivity_data_2class_pIC50.csv')



#Exploratory Data Analysis via Lipinski descriptors
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt
custom_palette = {'active': 'blue', 'inactive': 'red'}


#Frequency plot of active and inactive bioactivity classes
plt.figure(figsize=(5.5, 5.5))

sns.countplot(x='class', data=df_2class, edgecolor='black', palette=custom_palette)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

plt.savefig('plot_bioactivity_class.pdf')


#Scatter plot of MW and LogP 

plt.figure(figsize=(8, 8))

# Remove rows where MW, LogP, or pIC50 contain NaN or infinite values
df_2class = df_2class.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
df_2class = df_2class.dropna(subset=['MW', 'LogP', 'pIC50'])  # Drop rows with NaN

print(df_2class[['MW', 'LogP', 'pIC50']].describe())  # Check if data is clean


sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='class', size='pIC50', sizes=(20, 200), palette=custom_palette, edgecolors='black', alpha=0.7)

plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)

plt.savefig('plot_MW_vs_LogP.pdf', bbox_inches = "tight")  # Save plot as PDF

plt.show()  # Show the plot

#Box plots of pIC50 against activity class
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'class', y = 'pIC50', data = df_2class, palette=custom_palette)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')

plt.savefig('plot_ic50.pdf')

#Statistical analysis (Mann-Whitney U Test)#https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
from numpy.random import seed
from scipy.stats import mannwhitneyu

def mannwhitney(descriptor, verbose=False):
    # Seed the random number generator
    seed(1)

    # Select active and inactive classes based on descriptor
    selection = [descriptor, 'class']
    df = df_2class[selection]  # Assuming df_2class is defined globally

    active = df[df['class'] == 'active'][descriptor]
    inactive = df[df['class'] == 'inactive'][descriptor]

    # Perform Mann-Whitney U test
    stat, p = mannwhitneyu(active, inactive)

    # Interpretation of the result
    alpha = 0.05
    interpretation = "Same distribution (fail to reject H0)" if p > alpha else "Different distribution (reject H0)"

    # Store results in a DataFrame
    results = pd.DataFrame({
        'Descriptor': [descriptor],  # Wrap in list to prevent errors
        'Statistics': [stat],
        'p': [p],
        'alpha': [alpha],
        'Interpretation': [interpretation]
    })

    # Save results as CSV file
    filename = f"mannwhitneyu_{descriptor}.csv"
    results.to_csv(filename, index=False)

    if verbose:
        print(results)

    return results

#MannWhitney Test for pIC50
print(mannwhitney('pIC50', verbose=True))

#Box plot of Molecular Weight against bioactivity class
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'class', y = 'MW', data = df_2class, palette=custom_palette)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')

plt.savefig('plot_MW.pdf')

plt.show()

#MannWhitney Test for MW
print(mannwhitney('MW'))

#Box plot of LogP against bioactivity 
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'class', y = 'LogP', data = df_2class, palette=custom_palette)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')

plt.savefig('plot_LogP.pdf')

plt.show()

#MannWhitney Test for LogP
print(mannwhitney('LogP'))


#Box plot of NumHDonors against bioactivity 
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'class', y = 'NumHDonors', data = df_2class, palette=custom_palette)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')

plt.savefig('plot_NumHDonors.pdf')

plt.show()

#MannWhitney Test for NumHDonors
print(mannwhitney('NumHDonors'))


#Box plot of NumHAcceptors against bioactivity 
plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'class', y = 'NumHAcceptors', data = df_2class, palette=custom_palette)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')

plt.savefig('plot_NumHAcceptors.pdf')

plt.show()

#MannWhitney Test for NumHAcceptors
print(mannwhitney('NumHAcceptors'))
















