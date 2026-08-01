import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("spam_detc_data.csv", sep='\t')
print("\nDataset:")
print(df.head(15))

# Mapping features 
df['msg_len'] = df['num_characters']
df['word_count'] = df['num_words']

X = df[['msg_len', 'word_count']]
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


models = {
    'SVM': SVC(kernel='linear', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

model_names = []
accuracies = []

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    model_names.append(name)
    accuracies.append(acc)
    print(f"{name} Accuracy: {acc:.2f}%")

best_model=max(models.items(),key=lambda x:x[1].score(X_test,y_test))
print("\nBest Model:",best_model[0])

# Plot Comparison Bar Chart
plt.bar(model_names, accuracies, color=['lightblue', 'lightpink', 'lightgreen'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
plt.show()

