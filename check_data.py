import pandas as pd
df = pd.read_csv('Data/tasks.csv')
print('Categories in data:', df['category'].unique())
print('Priorities in data:', df['priority'].unique())
