import numpy as np
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from ucimlrepo import fetch_ucirepo
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Definizione della classe Naive_Bayes_Bernoulli
class Naive_Bayes_Bernoulli(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.models = []
        self.freq = []
        self.classes = np.unique(y)
        for c in self.classes:
            self.freq.append((y == c).sum() / y.shape[0])
            self.models.append(X[y == c].mean(axis=0))

    def predict(self, X):
        size = X.shape[0]
        y_pred = np.zeros(size, dtype=self.classes.dtype)
        probs = np.zeros(len(self.classes))
        for i in range(size):
            max_prob = 0
            max_c = 0
            for c in range(len(self.classes)):
                cond_P = (self.models[c] * (X[i] >= 0.5) + (1 - self.models[c]) * (X[i] < 0.5)).prod()
                probs[c] = cond_P * self.freq[c]
                if probs[c] > max_prob:
                    max_prob = probs[c]
                    max_c = c
            y_pred[i] = self.classes[max_c]
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Fetch dataset
spambase = fetch_ucirepo(id=94)

# Data (as pandas dataframes)
X = spambase.data.features.iloc[:, :54]
y = spambase.data.targets.values.ravel()  # Convert DataFrame to numpy array and flatten to 1D array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Trasformazione TF-IDF
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Standardizza i dati
scaler = StandardScaler(with_mean=False)  # with_mean=False per mantenere la matrice sparsa
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)

# Modelli
models = {
    'SVM (linear)': SVC(kernel='linear', C=1.0, random_state=1),
    'SVM (poly)': SVC(kernel='poly', degree=2, C=1.0, random_state=1),
    'SVM (rbf)': SVC(kernel='rbf', C=1.0, random_state=1),
    'Random Forest': RandomForestClassifier(random_state=1),
    'Naive Bayes': Naive_Bayes_Bernoulli(), #In alternativa si può usare la classe BernoulliNB di sklearn
    'k-NN': KNeighborsClassifier(n_neighbors=5)
}

# Cross-validation sul training set e fitting sul test set
for name, model in models.items():
    if name == 'Naive Bayes':
        X_train_toUse = X_train_scaled.toarray()
        X_test_toUse = X_test_scaled.toarray()
    else:
        X_train_toUse = X_train_scaled
        X_test_toUse = X_test_scaled
    
    print(f"-------- {name.upper()} --------")
    scores = cross_val_score(model, X_train_toUse, y_train, cv=10)
    print(f"Cross-Validation accuracy: {scores.mean():.4f}")
    start_time = time.time()
    model.fit(X_train_toUse, y_train)
    print(f"Training time: {time.time() - start_time:.4f}")
    start_time = time.time()
    y_pred_mod = model.predict(X_test_toUse)
    print('Prediction time: %f'%(time.time() - start_time))
    print('Missclassified examples: %d'% (y_test != y_pred_mod).sum())
    print('Test accuracy: %.3f' % accuracy_score(y_test,y_pred_mod))
    print()