# 81
# Wykorzystując klasę IsolationForest z pakietu scikit-learn dokonaj analizy elementów odstających na podanym zbiorze. Przekaż argumenty:
# n_estimators=100
# contamination=0.05
# random_state=42
# Dla przypomnienia 1 oznacza normalny element, -1 element odstający. Przypisz nową kolumnę do obiektu df o nazwie 'outlier_flag', 
# która będzie przechowywać informację czy dana próbka jest elementem normalnym czy odstającym. Wydrukuj dziesięć pierwszych wierszy obiektu df do konsoli.
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


np.random.seed(42)

df = pd.read_csv('F:/UdemyMachine2/factory.csv')

classifier = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

y_out = classifier.fit_predict(df)
df['outlier_flag'] = y_out
print(df.head(10))










# 82
# Zbadaj liczbę elementów odstających w zbiorze, tzn. zbadaj rozkład kolumny outlier_flag. Wynik wydrukuj do konsoli.
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


np.random.seed(42)

df = pd.read_csv('F:/UdemyMachine2/factory.csv')

classifier = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

y_out = classifier.fit_predict(df)
df['outlier_flag'] = y_out
print(df['outlier_flag'].value_counts())



# 83
# Wykorzystując funkcję load_digits() z pakietu scikit-learn załaduj dane dotyczące obrazów o rozdzielczości 8x8 pikseli do zmiennych: 
# data - obrazy zapisane w postaci tablic numpy kształtu (64,)
# target - etykiety, cyfry widoczne na obrazach
# Zapoznaj się dokładnie z podanym zbiorem. Spróbuj wyświetlić kilka przykładowych obrazów. 
# W celu wyświetlenia obrazu można użyć pakietu matplotlib następująco:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


np.random.seed(42)
data, target = load_digits(return_X_y=True)



idx = 250
plt.imshow(data[idx].reshape(8, 8), cmap='gray_r')
plt.title(f'Label: {target[idx]}')
plt.show()

print(target[250])






# 84
# Wykorzystując funkcję load_digits() z pakietu scikit-learn załadowano dane dotyczące obrazów o rozdzielczości 8x8 pikseli do zmiennych: 
# data - obrazy zapisane w postaci tablicy numpy o kształcie (1797, 64)
# target - etykiety, cyfry widoczne na obrazach w postaci tablicy numpy o kształcie (1797,)
# Dokonaj standaryzacji zmiennej data. Używając funkcji train_test_split() (ustaw argument random_state=42) podziel dane na zbiór treningowy i testowy:
# X_train, y_train
# X_test, y_test
# W odpowiedzi wyświetl kształty otrzymanych tablic tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


np.random.seed(42)
data, target = load_digits(return_X_y=True)



X_std = data / 255.

X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, random_state=42)


print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')






# 85
# Wykorzystując klasę KNeighborsClassifier z pakietu scikit-learn zbuduj model klasyfikacji wieloklasowej. 
# Wyucz model na danych treningowych i następnie dokonaj oceny na danych testowych. Dokładność modelu wyświetl do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


np.random.seed(42)
data, target = load_digits(return_X_y=True)
data = data / data.max()

X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state=42
)


classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
acc = classifier.score(X_test, y_test)

print(f'KNN accuracy: {acc:.4f}')





# 86
# Wykorzystując klasę LogisticRegression z pakietu scikit-learn zbuduj model klasyfikacji. Ustaw maksymalną liczbę iteracji w modelu na 100 - parametr max_iter.
# Wyucz model na danych treningowych i następnie dokonaj oceny na danych testowych. Dokładność modelu wyświetl do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


np.random.seed(42)
data, target = load_digits(return_X_y=True)
data = data / data.max()

X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state=42
)

lr = LogisticRegression(max_iter=100)
lr.fit(X_train, y_train)
acc = classifier.score(X_test, y_test)

print(f'Logistic Regression accuracy: {acc:.4f}')







# 87
# Podane są dwa pliki:
# data_train.csv
# target_train.csv
# Plik data_train.csv zawiera maile dotyczące dwóch kategorii: grafiki komputerowej (comp.graphics) oraz przestrzeni kosmicznej (sci.space). 
# Plik target_train.csv zawiera odpowiednio etykiety (0 - comp.graphics, 1 - sci.space). Wczytaj zawartość plików jako obiekty DataFrame odpowiednio o nazwach:
# data_train
# target_train
# Zapoznaj się z danymi. W odpowiedzi wydrukuj zawartość drugiego elementu obiektu data_train.
import numpy as np
import pandas as pd

data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')
print(data_train['text'][1])






# 88
# Przekształć obiekt data_train do postaci listy. Przypisz zmiany na trwałe do zmiennej data_train. W odpowiedzi wydrukuj długość listy data_train do konsoli.
import numpy as np
import pandas as pd

data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

data_train = data_train['text'].tolist()
print(len(data_train))








# 89
# Dokonano pewnego przekształcenia zmiennych data_train oraz target_trian. Wykorzystując klasę CountVectorizer z pakietu scikit-learn 
# dokonaj wektoryzacji tekstu znajdującego się w liście data_train i przypisz do zmiennej data_train_vectorized. 
# W odpowiedzi wydrukuj do konsoli kształt otrzymanej w ten sposób macierzy rzadkiej (sparse matrix).
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

cv = CountVectorizer()
cv.fit(data_train)
data_train_vectorized = cv.transform(data_train)
print(data_train_vectorized.shape)











# 90
# Wykorzystując klasę MultinomialNB zbuduj model klasyfikacji dokumentów tekstowych. Model wytrenuj w oparciu o dane data_train_vectorized oraz target_train.
# Następnie dokonaj klasyfikacji poniższych zdań:

# 'The graphic designer requires a good processor to work.'

# 'Flights into space.'

# Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


data_train = pd.read_csv(r'F:\UdemyMachine2\data_train.csv')
target_train = pd.read_csv(r'F:\UdemyMachine2\target_train.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()


cv = CountVectorizer()
data_train_vectorized = cv.fit_transform(data_train)    

classifier = MultinomialNB()
classifier.fit(data_train_vectorized, target_train)

docs = ['The graphic designer requires a good processor to work.', 'Flights into space.']

new_data = cv.transform(docs)

data_pred = classifier.predict(new_data)

for doc, pred in zip(docs, data_pred):
  print(f'\'{doc}\' => {categories[pred]}')