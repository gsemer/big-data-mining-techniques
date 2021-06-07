import pandas as pd
import seaborn as sns


df_categories = pd.read_csv('testSet_categories.csv')
sns.countplot(df_categories['Predicted'])


df_duplicate = pd.read_csv('duplicate_predictions.csv')
sns.countplot(df_duplicate['Predicted'])

