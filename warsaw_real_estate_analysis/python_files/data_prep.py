import pandas as pd
from sklearn.impute import KNNImputer

train_df = pd.read_csv("data/apartments_rent_pl_2023_11.csv")
train_df = train_df[train_df['city'].str.lower() == 'warszawa']
train_df = train_df.drop(columns=['city'])

train_df['isBrick'] = train_df['buildingMaterial'].apply(lambda x: 1 if x == 'brick' else 0)
train_df['isConcreteSlab'] = train_df['buildingMaterial'].apply(lambda x: 1 if x == 'concreteSlab' else 0)
train_df = train_df.drop(columns=['buildingMaterial'])

train_df['isPremium'] = train_df['condition'].apply(lambda x: 1 if x == 'premium' else 0)
train_df['isLowQual'] = train_df['condition'].apply(lambda x: 1 if x == 'low' else 0)
train_df = train_df.drop(columns=['condition'])

train_df['isApartment'] = train_df['type'].apply(lambda x: 1 if x == 'apartmentBuilding' else 0)
train_df['isBlock'] = train_df['type'].apply(lambda x: 1 if x == 'blockOfFlats' else 0)
train_df['isTenement'] = train_df['type'].apply(lambda x: 1 if x == 'tenement' else 0)
train_df = train_df.drop(columns=['type'])

train_df['hasElevator'] = train_df['hasElevator'].fillna('no')

knn_imputer = KNNImputer(n_neighbors=5)
columns_to_impute = ['buildYear', 'floorCount', 'floor'] + [col for col in train_df.columns if col.endswith('Distance')]
train_df[columns_to_impute] = knn_imputer.fit_transform(train_df[columns_to_impute])
train_df[columns_to_impute] = train_df[columns_to_impute].astype(int)

train_df['age'] = 2024 - train_df['buildYear']
train_df = train_df.drop(columns=['buildYear'])
train_df = train_df.drop(columns=['ownership'])
train_df.replace({'yes': 1, 'no': 0}, inplace=True)


test_df = pd.read_csv("apartments_rent_pl_2024_05.csv")
test_df = test_df[test_df['city'].str.lower() == 'warszawa']
test_df = test_df.drop(columns=['city'])


test_df['isBrick'] = test_df['buildingMaterial'].apply(lambda x: 1 if x == 'brick' else 0)
test_df['isConcreteSlab'] = test_df['buildingMaterial'].apply(lambda x: 1 if x == 'concreteSlab' else 0)
test_df = test_df.drop(columns=['buildingMaterial'])

test_df['isPremium'] = test_df['condition'].apply(lambda x: 1 if x == 'premium' else 0)
test_df['isLowQual'] = test_df['condition'].apply(lambda x: 1 if x == 'low' else 0)
test_df = test_df.drop(columns=['condition'])

test_df['isApartment'] = test_df['type'].apply(lambda x: 1 if x == 'apartmentBuilding' else 0)
test_df['isBlock'] = test_df['type'].apply(lambda x: 1 if x == 'blockOfFlats' else 0)
test_df['isTenement'] = test_df['type'].apply(lambda x: 1 if x == 'tenement' else 0)
test_df = test_df.drop(columns=['type'])

test_df['hasElevator'] = test_df['hasElevator'].fillna('no')

knn_imputer = KNNImputer(n_neighbors=5)
columns_to_impute = ['buildYear', 'floorCount', 'floor'] + [col for col in test_df.columns if col.endswith('Distance')]
test_df[columns_to_impute] = knn_imputer.fit_transform(test_df[columns_to_impute])
test_df[columns_to_impute] = test_df[columns_to_impute].astype(int)

test_df['age'] = 2024 - test_df['buildYear']
test_df = test_df.drop(columns=['buildYear'])
test_df = test_df.drop(columns=['ownership'])
test_df.replace({'yes': 1, 'no': 0}, inplace=True)
test_df = test_df[~test_df['id'].isin(train_df['id'])]

train_df.to_csv('trainset.csv', index=False)
test_df.to_csv('testset.csv', index=False)
