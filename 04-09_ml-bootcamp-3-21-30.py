# 21
# Zbuduj model regresji logistycznej (ustaw tylko argument max_iter=100, resztę pozostaw domyślnie) 
# wykorzystując pakiet scikit-learn oraz dane IRIS.
# Model wytrenuj na danych treningowych i następnie dokonaj oceny modelu na zbiorze testowym.
# W odpowiedzi wydrukuj dokładność modelu na zbiorze testowym tak jak pokazano poniżej.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=20)

lr = LogisticRegression(max_iter=100)
lr.fit(data_train, target_train)
accuracy = lr.score(data_test, target_test)

print(f'Accuracy: {accuracy:.4f}')






# 22
# Zbudowano model regresji logistycznej wykorzystując pakiet scikit-learn oraz dane IRIS. 
# Dokonaj predykcji danych testowych na podstawie modelu i przypisz do zmiennej target_pred. 
# W odpowiedzi wyświetl zmienną target_pred do konsoli.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.3, random_state=20
)

model = LogisticRegression(max_iter=1000)
model.fit(data_train, target_train)
target_pred = model.predict(data_test)
print(target_pred)






# 23
# Zbudowano model regresji logistycznej wykorzystując pakiet scikit-learn oraz dane IRIS. 
# Dokonano predykcji danych testowych na podstawie modelu i przypisano do zmiennej target_pred.
# Wyznacz macierz pomyłek (macierz konfuzji) i wydrukuj ją do konsoli.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.3, random_state=20
)

model = LogisticRegression(max_iter=1000)
model.fit(data_train, target_train)

target_pred = model.predict(data_test)

cm = confusion_matrix(target_test, target_pred)
print(cm)





# 24
# Zbudowano model regresji logistycznej wykorzystując pakiet scikit-learn oraz dane IRIS. 
# Dokonano predykcji danych testowych na podstawie modelu i przypisano do zmiennej target_pred.
# Wyświetl raport klasyfikacji modelu wykorzystując funkcję classification_report() z pakietu scikit-learn.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.3, random_state=20
)

model = LogisticRegression(max_iter=1000)
model.fit(data_train, target_train)

target_pred = model.predict(data_test)

cr = classification_report(target_test, target_pred)
print(cr)




# 25
# Wykorzystując klasę LabelEncoder z pakietu scikit-learn dokonaj kodowania 0-1 kolumny bought. Przypisz zmiany do obiektu df.
# W odpowiedzi wydrukuj obiekt df do konsoli.
import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes'],
}

df = pd.DataFrame(data=data)
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')

df

le = LabelEncoder()
le.fit(df['bought'])
df['bought'] = le.transform(df['bought'])
print(df)










# 26
# Wykorzystując klasę OneHotEncoder z pakietu scikit-learn dokonaj kodowania 0-1 kolumny size (ustaw argument sparse=False). 
# Wydrukuj zakodowaną postać kolumny size do konsoli (nie przypisuj zmian do zmiennej df). 
# Wydrukuj także otrzymane kategorie przy kodowaniu kolumny size tak jak pokazano poniżej.
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}

df = pd.DataFrame(data=data)
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')


encoder = OneHotEncoder(sparse_output=True)
coded = encoder.fit_transform(df[['size']])
print(coded)
print(encoder.categories_)







# 27
# Załaduj dane Breast Cancer Data wykorzystując funkcję load_breast_cancer() z pakietu scikit-learn do zmiennej raw_data. 
# Następnie wydrukuj informacje o tym zbiorze do konsoli (zawartość klucza 'DESCR').
from sklearn.datasets import load_breast_cancer


raw_data = load_breast_cancer()
print(raw_data.DESCR)






# 28
# Poniżej załadowano zbiór Breast Cancer Data do zmiennej raw_data.
# Przypisz do zmiennej data tablicę numpy z danymi ze zmiennej raw_data (zawartość klucza 'data') 
# oraz do zmiennej target tablicę numpy ze zmienną docelową (zawartość klucza 'target').
import numpy as np
from sklearn.datasets import load_breast_cancer


np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()
data = raw_data.data
target = raw_data.target

print(data)
print(target)









# 29
# Połącz te dwie tablice w jedną o nazwie all_data i wydrukuj trzy pierwsze wiersze tej tablicy do konsoli.
import numpy as np
from sklearn.datasets import load_breast_cancer


np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

all_data = np.c_[data, target]

print(all_data[:3])





# 30
# Połączono te dwie tablice w jedną o nazwie all_data. Utwórz z tablicy all_data obiekt DataFrame nadając odpowiednio nazwy kolumn 
# (zawartość klucza 'feature_names' obiektu raw_data + nazwa zmiennej docelowej jako 'target').
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

all_data = np.c_[data, target]

df = pd.DataFrame(data=all_data, columns=list(raw_data.feature_names) + ['target'])
print(df.head())
