#Import Libraries 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Load dataset
data = 'https://raw.githubusercontent.com/RebeccaRKS1998/Bio_Project/refs/heads/main/acetylecholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv'

df = pd.read_csv('acetylecholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')

#Input features 
X = df.drop('pIC50', axis=1)
print(X)

#Output features
Y = df.pIC50
print(Y)

#Examining data dimension
print(X.shape)
print(Y.shape)

#Remove low variance features
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = selection.fit_transform(X)
print(X.shape)

#Data split (80/20 ratio)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#Combine X_train and Y_train for joint filtering
train_df =pd.DataFrame(X_train)
train_df['pIC50'] = Y_train.reset_index(drop=True)

#Replace inf with NaN, then drop rows with NaN
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.dropna(inplace=True)

#Seperate back into X and Y
X_train_clean = train_df.drop('pIC50', axis=1)
Y_train_clean = train_df['pIC50']
print("Cleaned X_train shape:", X_train_clean.shape)
print("Cleaned Y_train shape:", Y_train_clean.shape)

#Build Regression Model using Random Forest
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train_clean, Y_train_clean)

# Evaluate model
r2 = model.score(X_test, Y_test)
print("R² score:", r2)

Y_pred = model.predict(X_test)

#Scatter plot of Experimental vs Predicted pIC50 Values
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
sns.set_style("white")

ax = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)
plt.show()