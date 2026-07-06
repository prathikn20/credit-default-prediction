import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import joblib


#Import dataset
path = "data/UCI_credit_default.xls"
df = pd.read_excel(path, header = 1)

#Preprocess data
df = df.drop(columns = ['ID'])
X = df.drop(columns = 'default payment next month')
Y = df['default payment next month']

#Split into test and training/validation data
X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

#Split into train and validation data
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)

#Categorical Handling
categorical_cols = ['EDUCATION', 'MARRIAGE', 'SEX']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

#Hyperparameter tuning
depths = [7, 8, 9, 10, 12, 15]
leaves = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

best_score = -1
best_depth = None
best_leaf = None

for depth in depths:
    for leaf in leaves:
        pipe = Pipeline([
        ('preprocessor', preprocessor),      
        ('classifier', RandomForestClassifier(n_estimators= 200,
                                          max_depth=depth, 
                                          min_samples_leaf=leaf,
                                          class_weight='balanced',
                                          random_state=42,
                                          n_jobs=-1))
        ])

        pipe.fit(X_train, y_train)

        score = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])

        if score > best_score:
            best_score = score
            best_depth = depth
            best_leaf = leaf

print(f"Best Score: {best_score}, Best Max Depth: {best_depth}, Best Min Samples Lead: {best_leaf}")

pipe = Pipeline([
        ('preprocessor', preprocessor),      
        ('classifier', RandomForestClassifier(n_estimators= 200,
                                          max_depth=best_depth, 
                                          min_samples_leaf=best_leaf,
                                          class_weight='balanced',
                                          random_state=42,
                                          n_jobs = -1))
        ])

pipe.fit(X_temp, y_temp)

AUC_score = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
y_pred = (pipe.predict_proba(X_test)[:, 1] >= 0.4).astype(int)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"AUC: {AUC_score:.3f}, Recall @ 0.4: {recall:.3f}, Precision @ 0.4: {precision:.3f}")

joblib.dump(pipe, "models/credit_pipeline.joblib")