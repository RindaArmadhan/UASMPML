import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Load dataset
data = pd.read_csv('restaurant_menu_optimization_data.csv')

# Encode categorical features
le_category = LabelEncoder()
le_item = LabelEncoder()
data['MenuCategory'] = le_category.fit_transform(data['MenuCategory'])
data['MenuItem'] = le_item.fit_transform(data['MenuItem'])

# Features and target
X = data[['MenuCategory', 'MenuItem', 'Price']]
y = data['Profitability']

# Encode target labels
le_profitability = LabelEncoder()
y = le_profitability.fit_transform(y)

# Train model
model = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=50,
    subsample=1.0
)
model.fit(X, y)

# Save the model and encoders
joblib.dump(model, 'model.sav')
joblib.dump(le_category, 'le_category.sav')
joblib.dump(le_item, 'le_item.sav')
joblib.dump(le_profitability, 'le_profitability.sav')