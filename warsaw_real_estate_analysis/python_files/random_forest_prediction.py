import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

train_file_path = 'trainset.csv'
test_file_path = 'testset.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

feature_columns = [col for col in train_df.columns if col not in ['price', 'id']]
target_column = 'price'

X_train = train_df[feature_columns]
y_train = train_df[target_column]
X_test = test_df[feature_columns]
y_test = test_df[target_column]

param_grid = {
    'n_estimators': [250, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

"""
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print(f'Best parameters found: {grid_search.best_params_}')
"""

model = RandomForestRegressor(max_depth = 20,max_features='sqrt', min_samples_leaf = 1, min_samples_split = 2, n_estimators=1000, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error (MAE): {mae:.3f}')

test_df['predicted_price'] = predictions

"""
plt.scatter(y_test, predictions, alpha=1, s= 8)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line for reference
plt.show()
"""

importance = model.feature_importances_
indices = importance.argsort()[-10:]  
"""
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
plt.xlabel('Feature Importance Score')
plt.title('Top 10 Feature Importances')
plt.show()
"""
top_5_features = [feature_columns[i] for i in indices][-5:]

for feature in top_5_features:
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = PartialDependenceDisplay.from_estimator(model, X_train, features=[feature], ax=ax)
    ax.set_ylabel('price')
    plt.title(f'Partial Dependence of {feature}')
    plt.show()