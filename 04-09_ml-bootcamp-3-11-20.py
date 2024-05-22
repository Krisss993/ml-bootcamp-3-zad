# 11
# Dokonaj kodowania 0-1 obiektu df (dokładnie kolumny weight_cut) dzięki funkcji pd.get_dummies(). W odpowiedzi wynik kodowania wydrukuj do konsoli.
import pandas as pd


df = pd.DataFrame(
    data={'weight': [75.0, 78.5, 85.0, 91.0, 84.5, 83.0, 68.0]}
)
df['weight_cut'] = pd.cut(
    df['weight'],
    bins=(60, 75, 80, 95),
    labels=['light', 'normal', 'heavy'],
)

df = pd.get_dummies(df, dtype='int')
print(df)




# 12
# Do obiektu df przypisz nową kolumnę o nazwie 'number', która przyjmie liczbę elementów listy w kolumnie currency. 
# W odpowiedzi wydrukuj obiekt DataFrame do konsoli.
import pandas as pd


data_dict = {
    'currency': [
        ['PLN', 'USD'],
        ['EUR', 'USD', 'PLN', 'CAD'],
        ['GBP'],
        ['JPY', 'CZK', 'HUF'],
        [],
    ]
}
df = pd.DataFrame(data=data_dict)

df['number'] = df['currency'].apply(lambda x:len(x))
print(df)



# 13
# Przypisz do obiektu df nową kolumnę o nazwie 'PLN_flag', która przyjmie wartość 1, 
# gdy waluta 'PLN' będzie w liście w kolumnie currency i przeciwnie 0. W odpowiedzi wydrukuj obiekt DataFrame do konsoli.
import pandas as pd


data_dict = {
    'currency': [
        ['PLN', 'USD'],
        ['EUR', 'USD', 'PLN', 'CAD'],
        ['GBP'],
        ['JPY', 'CZK', 'HUF'],
        [],
    ]
}
df = pd.DataFrame(data=data_dict)

df['PLN_flag'] = df['currency'].apply(lambda x:'PLN' in x).astype('int')
df





# 14
# Podziel wartości kolumny hashtags względem znaku hash '#' używając pd.Series.str.split() z argumentem expand=True. Otrzymasz cztery kolumny.
import pandas as pd


df = pd.DataFrame(
    data={
        'hashtags': [
            '#good#vibes',
            '#hot#summer#holiday',
            '#street#food',
            '#workout',
        ]
    }
)


df = df['hashtags'].str.split('#', expand=True)
df = df.drop(0, axis=1)
df.columns = ['hashtag1', 'hashtag2', 'hashtag3']
print(df)







# 15
# Utwórz nową kolumnę o nazwie 'missing' w obiekcie df i przypisz do niej liczbę brakujących hashtagów dla każdego wiersza.
# Przykładowo, wiersz pierwszy -> 1, wiersz drugi -> 0, wiersz trzeci -> 1, itd.
import pandas as pd


df = pd.DataFrame(
    data={
        'hashtags': [
            '#good#vibes',
            '#hot#summer#holiday',
            '#street#food',
            '#workout',
        ]
    }
)

df = df['hashtags'].str.split('#', expand=True)
df = df.drop(columns=[0])
df['missing'] = df.isna().sum(axis=1)
df





# 16
# Przygotuj kolumnę investments do modelu, tzn. przekształć ją odpowiednio na typ int.
# W odpowiedzi wydrukuj obiekt DataFrame do konsoli.
import pandas as pd


df = pd.DataFrame(
    data={
        'investments': [
            '100_000_000',
            '100_000',
            '30_000_000',
            '100_500_000',
        ]
    }
)

df['investments'] = df['investments'].astype('int')
print(df)






# 17
# Załaduj zbiór danych IRIS do zmiennej data wykorzystując pakiet scikit-learn oraz funkcję load_iris(). 
# Następnie wyświetl wszystkie klucze zmiennej data do konsoli.
from sklearn.datasets import load_iris


data = load_iris()
print(data.keys())







# 18
# Załadowano zbiór IRIS wykorzystując pakiet scikit-learn do zmiennej data.
# Wyświetl nazwy zmiennych (klucz 'feature_names') oraz nazwy klas (klucz 'target_names') w zbiorze IRIS tak jak pokazano poniżej.
from sklearn.datasets import load_iris


data = load_iris()
print(data.feature_names)
print(data['target_names'])





# 19
# Poniżej załadowano zbiór IRIS wykorzystując pakiet scikit-learn do zmiennej data_raw.
# Do zmiennej data przypisz dane zbioru IRIS (klucz 'data'). 
# Do zmiennej target przypisz wartości zmiennej docelowej (klucz 'target') ze zbioru IRIS.
# W odpowiedzi wydrukuj kształt zmiennych: data oraz target do konsoli.
from sklearn.datasets import load_iris


data_raw = load_iris()
data = data_raw.data
target = data_raw.target
print(data.shape)
print(target.shape)





# 20
# Wykorzystując pakiet scikit-learn oraz funkcję train_test_split() podziel dane na zbiór treningowy 
# (data_train, target_train) i testowy (data_test, target_test). Ustal rozmiar zbioru testowego na 30% próbek.
# W odpowiedzi wyświetl rozmiar poniższych tablic (tak jak pokazano poniżej):
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
print(f'data_train shape: {X_train.shape}')
print(f'target_train shape: {y_train.shape}')
print(f'data_test shape: {X_test.shape}')
print(f'target_test shape: {y_test.shape}')
















