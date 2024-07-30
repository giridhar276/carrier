# Import necessary libraries
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# Import Data
df = pd.read_csv('Ecom_Cust_Survey.csv', header=0)

# Drop missing values
df.dropna(inplace=True)

# Q1. How many customers have participated in the survey?
print("Number of customers:", df.shape[0])

# Q2. Overall most of the customers are satisfied or dissatisfied?
print(df['Overall_Satisfaction'].value_counts())

# Q3. Segment the data and find the concentrated satisfied and dissatisfied customer segments
# Map categorical data to numerical data
df['Region'] = df['Region'].map({'EAST': 1, 'WEST': 2, 'NORTH': 3, 'SOUTH': 4}).astype(int)
df['Customer_Type'] = df['Customer_Type'].map({'Prime': 1, 'Non_Prime': 0}).astype(int)
df.rename(columns={'Order Quantity': 'Order_Quantity', 'Improvement Area': 'Improvement_Area'}, inplace=True)
df['Improvement_Area'] = df['Improvement_Area'].map({'Website UI': 1, 'Packing & Shipping': 2, 'Product Quality': 3}).astype(int)
df['Overall_Satisfaction'] = df['Overall_Satisfaction'].map({'Dis Satisfied': 0, 'Satisfied': 1}).astype(int)

# Define features and labels
features = list(df.columns[2:6])
X = df[features]
y = df['Overall_Satisfaction']

# Build Tree Model
clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)

# Predict
predict1 = clf.predict(X)

# Confusion Matrix and Accuracy
cm = confusion_matrix(y, predict1)
print("Confusion Matrix:\n", cm)
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
print("Accuracy:", accuracy)

# Import and preprocess training and test data for overfitting lab
train = pd.read_csv("Train_data.csv", header=0)
test = pd.read_csv("Test_data.csv", header=0)

train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0}).astype(int)
train['Bought'] = train['Bought'].map({'Yes': 1, 'No': 0}).astype(int)
test['Gender'] = test['Gender'].map({'Male': 1, 'Female': 0}).astype(int)
test['Bought'] = test['Bought'].map({'Yes': 1, 'No': 0}).astype(int)

# Define features and labels for training and test data
features = list(train.columns[:2])
X_train = train[features]
y_train = train['Bought']
X_test = test[features]
y_test = test['Bought']

# Train Tree Model
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and calculate accuracy for train and test data
predict1 = clf.predict(X_train)
predict2 = clf.predict(X_test)

cm1 = confusion_matrix(y_train, predict1)
accuracy1 = (cm1[0,0] + cm1[1,1]) / cm1.sum()
print("Train Accuracy:", accuracy1)

cm2 = confusion_matrix(y_test, predict2)
accuracy2 = (cm2[0,0] + cm2[1,1]) / cm2.sum()
print("Test Accuracy:", accuracy2)

# Pruning
dtree = tree.DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=5, max_depth=5)
dtree.fit(X_train, y_train)
predict3 = dtree.predict(X_train)
predict4 = dtree.predict(X_test)

cm1 = confusion_matrix(y_train, predict3)
accuracy1 = (cm1[0,0] + cm1[1,1]) / cm1.sum()
print("Pruned Train Accuracy:", accuracy1)

cm2 = confusion_matrix(y_test, predict4)
accuracy2 = (cm2[0,0] + cm2[1,1]) / cm2.sum()
print("Pruned Test Accuracy:", accuracy2)

# Tree Building & Model Selection with Fiberbits dataset
Fiber_df = pd.read_csv("Fiberbits.csv", header=0)

features = list(Fiber_df.drop(['active_cust'], axis=1).columns)
X = Fiber_df[features].values
y = Fiber_df['active_cust'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_train, y_train)
print("Fiberbits Train Accuracy:", clf1.score(X_train, y_train))
print("Fiberbits Test Accuracy:", clf1.score(X_test, y_test))

# Optimal hyperparameters
tuned_parameters = [{'criterion': ['gini', 'entropy'], 'max_depth': range(2, 10)}]
clf_tree = tree.DecisionTreeClassifier()
clf = GridSearchCV(clf_tree, tuned_parameters, cv=10, scoring='roc_auc')
clf.fit(X_train, y_train)

print("Best Score:", clf.best_score_)
print("Best Parameters:", clf.best_params_)
