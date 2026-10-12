import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

data = {
    'Study_hours': [2, 4, 6, 8, 10],
    'Attendance': [60, 70, 80, 90, 100],
    'Previous_Marks': [55, 65, 75, 85, 95],
    'Assignment_Marks':[50,60,70,80,90]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
print("Normalised Data:")
print(normalized_df)

scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)

standardized_df = pd.DataFrame(standardized_data, columns=df.columns)

print(standardized_df)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(standardized_data)

pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])

print(pca_df)
