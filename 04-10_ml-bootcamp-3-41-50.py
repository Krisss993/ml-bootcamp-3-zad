


##################################




# 50
"""Wykorzystując klasę DecisionTreeClassifier z pakietu scikit-learn zbuduj model klasyfikacji dla podanych danych. Wykorzystując metodę przeszukiwania siatki oraz klasę GridSearchCV (ustaw argumentyscoring='accuracy', cv=5) znajdź optymalne wartości parametrów max_depth oraz min_samples_leaf. Wartości parametrów przeszukaj z podanych poniżej:

dla max_depth -> np.arange(1, 10)

dla min_samples_leaf -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

Dokonaj trenowania na zbiorze treningowym oraz oceny na zbiorze testowym.

W odpowiedzi wydrukuj do konsoli najbardziej optymalne wartości parametrów max_depth oraz min_samples_leaf."""

import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)
classifier = DecisionTreeClassifier()

params = {'max_depth': np.arange(1, 10), 'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}

grid = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=5)

grid.fit(X_train, y_train)
print(grid.best_params_
)

