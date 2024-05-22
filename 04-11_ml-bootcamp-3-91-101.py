# 91
# Dokonano pewnego przekształcenia zmiennych data_train oraz target_train. Wykorzystując klasę TfidfVectorizer z pakietu scikit-learn dokonaj 
# wektoryzacji tekstu znajdującego się w liście data_train i przypisz do zmiennej data_train_vectorized.
# W odpowiedzi wydrukuj do konsoli kształt otrzymanej w ten sposób macierzy rzadkiej (sparse matrix).
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

tfidf = TfidfVectorizer()
data_train_vectorized = tfidf.fit_transform(data_train)
print(data_train_vectorized.shape)








# 92
# Wykorzystując klasę MultinomialNB zbuduj model klasyfikacji dokumentów tekstowych. Model wytrenuj w oparciu o dane data_train_vectorized oraz target_train.
# Następnie dokonaj klasyfikacji poniższych zdań:

# 'The graphic designer requires a good processor to work.'
# 'Flights into space.'

# Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


data_train = pd.read_csv('data_train.csv')
target_train = pd.read_csv('target_train.csv')

categories = ['comp.graphics', 'sci.space']

data_train = data_train['text'].tolist()
target_train = target_train.values.ravel()

vectorizer = TfidfVectorizer()
data_train_vectorized = vectorizer.fit_transform(data_train)


docs = [
        'The graphic designer requires a good processor to work',
        'Flights into space'
        ]
new_data = vectorizer.transform(docs)

classifier = MultinomialNB()
classifier.fit(data_train_vectorized, target_train)
data_pred = classifier.predict(new_data)

for doc, pred in zip(docs, data_pred):
    print(f'\'{doc}\' => {categories[pred]}')
    
    
    
    



# 93
# Ustaw opcje pakietu pandas pozwalające na wyświetlenie 15 kolumn obiektu DataFrame oraz wyświetlenie długości linii składającej się ze 150 znaków.
# Następnie wykorzystując funkcję load_boston() z pakietu scikit-learn załaduj dane do zmiennej raw_data.
# W oparciu o klucze 'data' oraz 'target' zmiennej raw_data przygotuj poniższy obiekt DataFrame:
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()
 
df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)
print(df.head())







# 94
# Wyświetl korelację zmiennych ze zmienną docelową target (w kolejności malejącej). Wynik wydrukuj do konsoli tak jak pokazano poniżej.

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

c = raw_df.corr()[10].sort_values(ascending=False)
c





# 95
# Skopiuj obiekt df do zmiennej data. Następnie wyrwij kolumnę target ze zmiennej data i przypisz do zmiennej target. 
# Wyświetl pięć pierwszych wierszy obiektu data, następnie wydrukuj pustą linię i kolejno pięć pierwszych wierszy obiektu target tak jak pokazano poniżej.
import numpy as np
import pandas as pd
 
from sklearn.datasets import load_boston
 
 
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()
 
df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)
 
data = df.copy()
target = data.pop('target')
 
print(data.head())
print()
print(target.head())









# 96
# Wykorzystując funkcję train_test_split() podziel dane (data, target) na zbiór treningowy i testowy (użyj argumentu random_datate=42) i 
# przypisz odpowiednio do zmiennych:
# data_train, target_train
# data_test, target_test
# W odpowiedzi wyświetl kształty obiektów: data_train, target_train, data_test, target_test tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)

data = df.copy()
target = data.pop('target')


data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)

print(f'data_train shape: {data_train.shape}')
print(f'target_train shape: {target_train.shape}')
print(f'data_test shape: {data_test.shape}')
print(f'target_test shape: {target_test.shape}')






# 97
# Wykorzystując klasę LinearRegression (z domyślnymi parametrami) z pakietu scikit-learn zbuduj model regresji liniowej. 
# Wyucz model na danych treningowych i dokonaj oceny na danych testowych. Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

lr = LinearRegression()
lr.fit(data_train, target_train)
acc = lr.score(data_test, target_test)
print(f'R^2 score: {acc:.4f}')







# 98
# Wykorzystano klasę LinearRegression z pakietu scikit-learn do zbudowania modelu regresji liniowej. 
# Wyuczono model na danych treningowych. Dokonaj predykcji na podstawie modelu na danych testowych i wynik przypisz do zmiennej target_pred.
# Wydrukuj zmienną target_pred do konsoli.
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

regressor = LinearRegression()
regressor.fit(data_train, target_train)

target_pred = regressor.predict(data_test)
print(target_pred)





# 99
# Wykorzystano klasę LinearRegression z pakietu scikit-learn do zbudowania modelu regresji liniowej. Wyuczono model na danych treningowych.
# Dokonano predykcji na podstawie modelu na danych testowych i wynik przypisano do zmiennej target_pred. 
# Zbuduj nowy obiekt DataFrame o nazwie predictions, który będzie przechowywał cztery kolumny:
# target_test
# target_pred
# error (różnica pomiędzy target_pred oraz target_test)
# abs_error (wartość bezwzględna z kolumny error)
# W odpowiedzi wydrukuj dziesięć pierwszych wierszy obiektu predictions do konsoli.
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

regressor = LinearRegression()
regressor.fit(data_train, target_train)

target_pred = regressor.predict(data_test)

predictions = pd.DataFrame({'target_test':target_test, 'target_pred':target_pred})
predictions['error'] = (predictions['target_pred']-predictions['target_test'])
predictions['abs_error'] = abs(predictions['error'])
print(predictions.head(10))





# 100
# Wykorzystując klasę GradientBoostingRegressor (z parametrem random_state=42) z pakietu scikit-learn zbuduj model regresji. 
# Wyucz model na danych treningowych i dokonaj oceny na danych testowych. Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

classifier = GradientBoostingRegressor(random_state=42)
classifier.fit(data_train, target_train)
acc = classifier.score(data_test, target_test)
print(f'R^2 score: {acc:.4f}')






# 101
# Wykorzystano klasę GradientBoostingRegressor z pakietu scikit-learn do zbudowania model regresji. Wyuczono model na danych treningowych. 
# Zapisz model (zmienna regressor) do pliku o nazwie 'model.pkl' wykorzystując moduł pickle.
# Następnie wczytaj plik model.pkl do zmiennej regressor_loaded. W odpowiedzi wydrukuj do konsoli informacje o obiekcie regressor_loaded wykonując poniższy kod:
import pickle

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

regressor = GradientBoostingRegressor()
regressor.fit(data_train, target_train)




with open('model.pkl', 'wb') as file:
    pickle.dump(regressor, file)
 
with open('model.pkl', 'rb') as file:
    regressor_loaded = pickle.load(file)
 
print(regressor_loaded)



