# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn tools.
2. Load the Iris dataset and create a DataFrame with features and target.
3. Separate features (x) and labels (y), then split into training and testing sets.
4. Create and train an SGDClassifier on the training data.
5. Use the trained model to predict labels on the test data.
6. Calculate and print the accuracy of the model.
7. Generate and print the confusion matrix to assess classification performance.
8. Plot the true vs. predicted labels to visualize prediction distribution.

## Program:


Program to implement the prediction of iris species using SGD Classifier.
Developed by: THAMIZH KUMARAN S
RegisterNumber:  212223240166
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

```

## Output:
Preview datasets :

![Screenshot 2025-04-28 160758](https://github.com/user-attachments/assets/53cf83b7-1949-46df-be90-f936b51de19c)

Classifier:

![Screenshot 2025-04-28 160806](https://github.com/user-attachments/assets/137b8554-a912-4819-b641-00f089af9cb9)

Accuracy :

![Screenshot 2025-04-28 160816](https://github.com/user-attachments/assets/687ca885-e527-4fcf-97f3-4d6bbc1980f3)

Confusion Matrix :

![Screenshot 2025-04-28 160825](https://github.com/user-attachments/assets/1f6adf4f-3724-48b4-a2f6-8c98f11a0f26)

Classification Report :

![Screenshot 2025-04-28 160835](https://github.com/user-attachments/assets/b1668ddc-e4a3-47fe-ae4a-669c69e96ddd)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
