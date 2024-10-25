# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_excel('heart_disease.xlsx')  # Ensure your file path is correct

# Remove any leading or trailing spaces from column names
data.columns = data.columns.str.strip()

# Print the column names for debugging
print("Column Names:\n", data.columns.tolist())

# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:\n", data.head())

# Check data types of each column
print("\nData Types:\n", data.dtypes)

# Check for missing values
print("\nMissing values:\n", data.isnull().sum())

# Check the shape of the DataFrame
print("\nShape of the DataFrame:\n", data.shape)

# Convert categorical columns to numeric (if needed)
# Adjust the following lines based on the actual column names you see
if 'sex' in data.columns:
    data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
else:
    print("'sex' column not found!")

if 'cp' in data.columns:
    data['cp'] = data['cp'].map({'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4})
else:
    print("'cp' column not found!")

if 'restecg' in data.columns:
    data['restecg'] = data['restecg'].map({'normal': 0, 'lv hypertrophy': 1, 'st-t abnormality': 2})
else:
    print("'restecg' column not found!")

if 'slope' in data.columns:
    data['slope'] = data['slope'].map({'upsloping': 1, 'flat': 2, 'downsloping': 3})
else:
    print("'slope' column not found!")

if 'thal' in data.columns:
    data['thal'] = data['thal'].map({'fixed defect': 1, 'normal': 2, 'reversible defect': 3})
else:
    print("'thal' column not found!")

# Ensure that all relevant columns are numeric
# Get a list of numeric columns after mapping categorical variables
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print the updated DataFrame and numeric columns for debugging
print("\nUpdated DataFrame after conversion:\n", data)
print("\nNumeric Columns:\n", numeric_cols)

# Check if the target column exists
target_column = 'num'  # Change this if the target column has a different name
if target_column not in data.columns:
    print(f"Target column '{target_column}' not found in DataFrame.")
else:
    # Visualize the distribution of numerical features
    if numeric_cols:
        data[numeric_cols].hist(bins=30, figsize=(15, 10))
        plt.show()

        # Box plots for detecting outliers in numeric features
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=data[numeric_cols])
        plt.show()

        # Correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.show()

        # Split the dataset
        X = data.drop(target_column, axis=1)  # Features
        y = data[target_column]  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Decision Tree Classifier
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X_train, y_train)

        # Make predictions and evaluate the model
        y_pred = dt_classifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

        # Hyperparameter Tuning
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }

        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        print("Best parameters found: ", grid_search.best_params_)

        # Use the best model from grid search and evaluate
        best_model = grid_search.best_estimator_
        y_pred_best = best_model.predict(X_test)
        print(classification_report(y_test, y_pred_best))
        print("Accuracy:", accuracy_score(y_test, y_pred_best))

        # Visualize the decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(best_model, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
        plt.show()
