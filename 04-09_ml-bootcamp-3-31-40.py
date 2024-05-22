# 31
# Wykorzystując funkcję train_test_split() z pakietu scikit-learn podziel dane (data oraz target) na zbiór treningowy i testowy, odpowiednio:
# zbiór treningowy: X_train, y_train
# zbiór testowy: X_test, y_test
# Ustaw argument random_state=40 oraz rozmiar zbioru testowego na 25%.
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=40) 
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')







# 32
# Sprawdź procentowy rozkład wartości zmiennych target, y_train oraz y_test. Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=40, test_size=0.25)

print(f'target :{np.unique(target, return_counts=True)[1]/len(target)}')
print(f'y_train:{np.unique(y_train, return_counts=True)[1]/len(y_train)}')
print(f'y_test :{np.unique(y_test, return_counts=True)[1]/len(y_test)}')






# 33
# Podziel dane (data, target) na zbiór treningowy i testowy, odpowiednio:
# zbiór treningowy: X_train, y_train
# zbiór testowy: X_test, y_test
# tak, aby zachować procentowy rozkład wartości w tablicach y_train oraz y_test taki jak w tablicy target.
# Następnie sprawdź procentowy rozkład wartości zmiennych target, y_train oraz y_test. 
# Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, stratify=target)
print(f'target :{np.unique(target, return_counts=True)[1]/len(target)}')
print(f'y_train:{np.unique(y_train, return_counts=True)[1]/len(y_train)}')
print(f'y_test :{np.unique(y_test, return_counts=True)[1]/len(y_test)}')





# 34
# Pierwsza kolumna opisuje lata pracy (zmienna objaśniająca), druga kolumna opisuje wynagrodzenie pracownika (zmienna objaśniana). 
# Wykorzystując równanie normalne oraz pakiet numpy znajdź równanie regresji liniowej. 
# Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd


df = pd.DataFrame(
    {
        'years': [1, 2, 3, 4, 5, 6],
        'salary': [4000, 4250, 4500, 4750, 5000, 5250],
    }
)

df
X1 = df.iloc[:,0].values.reshape(-1,1)
bias = np.ones((len(df), 1))
X = np.append(bias, X1, axis=1)
XT = X.reshape(2,-1)
XT
X_1 = np.linalg.inv(np.dot(XT,X))

X_2 = np.dot(XT, df.iloc[:,1].values.reshape(-1,1))

coefs = np.dot(X_1, X_2)
print(f'Linear regression: {coefs[0][0]:.2f} + {coefs[1][0]:.2f}x')









# 35
# Pierwsza kolumna opisuje lata pracy (zmienna objaśniająca), druga kolumna opisuje wynagrodzenie pracownika (zmienna objaśniana).
# Wykorzystując pakiet scikit-learn oraz klasę LinearRegression znajdź równanie regresji liniowej dla tego problemu. 
# Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.DataFrame(
    {
        'years': [1, 2, 3, 4, 5, 6],
        'salary': [4000, 4250, 4500, 4750, 5000, 5250],
    }
)


lr = LinearRegression()
lr.fit(df.iloc[:,0].values.reshape(-1, 1), df.iloc[:,1].values.reshape(-1, 1))
lr.coef_
lr.intercept_
print(
    f'Linear regression: {lr.intercept_[0]:.2f} + '
    f'{lr.coef_[0][0]:.2f}x')




# 36 
# Wczytaj dane z pliku data.csv do obiektu DataFrame. Następnie w oparciu o zmienną variable zbuduj model regresji liniowej 
# pozwalający przewidywać wartości zmiennej docelowej target (model zbuduj na wszystkich dostępnych danych). 
# Wykorzystaj w tym celu pakiet scikit-learn oraz klasę LinearRegression.
# Dokonaj oceny modelu wykorzystując score(). Wynik wydrukuj do konsoli (do czwartego miejsca po przecinku)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('data.csv')
data = df.variable.values.reshape(-1, 1)
target = df.target.values.reshape(-1, 1)

lr = LinearRegression()
lr.fit(data, target)
print(f'{lr.score(data, target):.4f}')






# 37
# Wczytaj plik data.csv do obiektu DataFrame. Następnie dokonaj ekstrakcji cech wielomianowych ze zmiennej var1 stopnia drugiego. 
# Otrzymane cechy w postaci tablicy numpy wyświetl do konsoli.
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


np.set_printoptions(suppress=True, precision=3)

df = pd.read_csv('data.csv')
poly = PolynomialFeatures(degree=2)
df_poly = poly.fit_transform(df.var1.values.reshape(-1,1))
print(df_poly)





# 38
# Wczytaj plik data.csv do obiektu DataFrame. 
# Następnie dokonaj ekstrakcji cech wielomianowych ze zmiennych var1 oraz var2 stopnia trzeciego. 
# Otrzymane cechy w postaci tablicy numpy wyświetl do konsoli.
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


np.set_printoptions(suppress=True, precision=3, linewidth=150)
df = pd.read_csv('data.csv')
poly = PolynomialFeatures(degree=3)
df_poly = poly.fit_transform(df)
print(df_poly)






# 39
# Plik predictions.csv zawiera predykcje pewnego modelu regresji:
# zmienna y_true opisuje rzeczywiste, zaobserwowane wartości
# zmienna y_pred opisuje wartości przewidziane przez model
# Wczytaj ten plik do obiektu DataFrame. Następnie zaimplementuj funkcję o nazwie mean_absolute_error() 
# obliczającą średni błąd bezwzględny predykcji.
# Wykorzystując zaimplementowaną funkcję policz wartość MAE dla tego modelu. Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

df = pd.read_csv('predictions.csv')

def mean_absolute_error():
    return abs(df['y_true'] - df['y_pred']).sum() / len(df['y_true'])

print(f'MAE = {mean_absolute_error():.4f}')








# 40
# Wczytaj ten plik do obiektu DataFrame. Następnie zaimplementuj funkcję o nazwie mean_squared_error() 
# obliczającą błąd średniokwadratowy predykcji.
# Wykorzystując zaimplementowaną funkcję policz wartość MSE dla tego modelu. Wynik wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd

df = pd.read_csv('predictions.csv')

def mean_squared_error():
    return ((df['y_true'] - df['y_pred'])**2).sum() / len(df['y_true'])

print(f'MSE = {mean_squared_error():.4f}')




