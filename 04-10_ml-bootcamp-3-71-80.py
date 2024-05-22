# 71
# Wczytaj plik pca.csv do obiektu DataFrame df. Plik zawiera zmienne var_1, ..., var_10. 
# Dokonaj analizy PCA z trzema komponentami wykorzystując pakiet scikit-learn oraz klasę PCA.
# W odpowiedzi wydrukuj procent wyjaśnionej wariancji przez te komponenty tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('F:/UdemyMachine2/pca.csv')

X = df.copy()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=3)
pca.fit(X_std)

res = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
res['cumulative'] = np.cumsum(res['explained_variance_ratio'])
res['component'] = res.index+1
print(res)







# 72
# Wczytaj plik pca.csv do obiektu DataFrame df. Plik zawiera zmienne var_1, ..., var_10. 
# Dokonaj analizy PCA wykorzystując pakiet scikit-learn. Zachowaj liczbę komponentów pozwalającą wyjaśnić 95% wariancji podanych danych. 
# W odpowiedzi podaj liczbę komponentów uzyskanych w analizie.
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv('F:/UdemyMachine2/pca.csv')

X = df.copy()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
pca.fit(X_std)

print(pca.n_components_)






# 73
# Każdy wiersz zawiera produkty zakupione przez jednego klienta. Podziel każdy wiersz kolumny products względem znaku spacji i rozszerz do obiektu DataFrame.
# Obiekt docelowo będzie posiadał 4 kolumny (maksymalna liczba produktów w jednej transakcji). W brakujące miejsca wpisz wartość None, 
# tak jak pokazano poniżej i przypisz do zmiennej expanded.
# W odpowiedzi wydrukuj zmienną expanded do konsoli.
import numpy as np
import pandas as pd


data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))

df = pd.DataFrame(data)
expanded = df['products'].str.split(' ', expand=True)
expanded.index = expanded.index+1
print(expanded)






# 74
# Do zmiennej products przypisz unikalne nazwy wszystkich produktów występujących w bazie transakcji posortowanych alfabetycznie. 
# Wydrukuj zmienną products do konsoli.
import numpy as np
import pandas as pd


data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for i in range(expanded.shape[0]):
    for j in range(expanded.shape[1]):
        if expanded.iloc[i,j] not in products and expanded.iloc[i,j]:
            products.append(expanded.iloc[i,j])
products.sort()            
print(products)




# 75
# Dokonaj kodowania 0-1 transakcji tak jak pokazano poniżej i przypisz do zmiennej transactions_encoded_df.
# Zmienną (obiekt DataFrame) transactions_encoded_df wydrukuj do konsoli.
import numpy as np
import pandas as pd


data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for col in expanded.columns:
    for product in expanded[col].unique():
        if product is not None and product not in products:
            products.append(product)
products.sort()


zeroes = np.zeros((len(expanded), len(products)),dtype='int8')

for (idx,product) in enumerate(products):
    for i in range(expanded.shape[0]):
        for j in range(expanded.shape[1]):
            if expanded.iloc[i,j] == product:
                zeroes[i,idx] = 1

transactions_encoded_df = pd.DataFrame(zeroes, columns=products)
print(transactions_encoded_df)





# 76
# Oblicz wsparcie (support) dla pojedynczych produktów i wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd


data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for col in expanded.columns:
    for product in expanded[col].unique():
        if product is not None and product not in products:
            products.append(product)
products.sort()

transactions_encoded = np.zeros(
    (len(transactions), len(products)), dtype='int8'
)

for row in zip(
    range(len(transactions)), transactions_encoded, expanded.values
):
    for idx, product in enumerate(products):
        if product in row[2]:
            transactions_encoded[row[0], idx] = 1
transactions_encoded_df = pd.DataFrame(
    transactions_encoded, columns=products
)

transactions_encoded_df

sups = []
for i in range(transactions_encoded_df.shape[1]):
    sups.append(round(transactions_encoded_df.iloc[:,i].sum()/transactions_encoded_df.shape[0],6))
sups
sup_ser = pd.Series(index=products, data=sups)
print(sup_ser)






# 77
# Oblicz wsparcie dla par: (butter, bread) oraz (butter, milk). Wynik wydrukuj do konsoli (zaokrąglij wsparcie do czterech miejsc po przecinku).
import numpy as np
import pandas as pd


data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for col in expanded.columns:
    for product in expanded[col].unique():
        if product is not None and product not in products:
            products.append(product)
products.sort()

transactions_encoded = np.zeros(
    (len(transactions), len(products)), dtype='int8'
)

for row in zip(
    range(len(transactions)), transactions_encoded, expanded.values
):
    for idx, product in enumerate(products):
        if product in row[2]:
            transactions_encoded[row[0], idx] = 1
transactions_encoded_df = pd.DataFrame(
    transactions_encoded, columns=products
)

transactions_encoded_df

sup_butter_bread = len(transactions_encoded_df.query("bread == 1 & butter == 1"))/len(transactions_encoded_df)
sup_butter_milk = len(transactions_encoded_df.query("butter == 1 & milk == 1"))/len(transactions_encoded_df)


print(f'support(butter, bread) = {sup_butter_bread:.4f}')
print(f'support(butter, milk) = {sup_butter_milk:.4f}')











# 78 
# Oblicz pewność reguł:
# cheese -> bread
# butter -> cheese
# Wynik wydrukuj do konsoli (zaokrąglij pewność do czterech miejsc po przecinku) tak jak pokazano poniżej.
import numpy as np
import pandas as pd


data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(expand=True)

products = []
for col in expanded.columns:
    for product in expanded[col].unique():
        if product is not None and product not in products:
            products.append(product)
products.sort()

transactions_encoded = np.zeros(
    (len(transactions), len(products)), dtype='int8'
)

for row in zip(
    range(len(transactions)), transactions_encoded, expanded.values
):
    for idx, product in enumerate(products):
        if product in row[2]:
            transactions_encoded[row[0], idx] = 1
transactions_encoded_df = pd.DataFrame(
    transactions_encoded, columns=products
)

conf_cheese_bread = len(transactions_encoded_df.query("cheese == 1 & bread == 1")) / transactions_encoded_df['cheese'].sum()
conf_butter_cheese = len(transactions_encoded_df.query("butter == 1 & cheese == 1")) / transactions_encoded_df['butter'].sum()

print(f'conf(cheese, bread) = {conf_cheese_bread:.4f}')
print(f'conf(butter, cheese) = {conf_butter_cheese:.4f}')





# 79
# Wykorzystując klasę LocalOutlierFactor z pakietu scikit-learn dokonaj analizy elementów odstających w podanym zbiorze. Ustaw argument:
# n_neighbors=20
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


np.random.seed(42)

df = pd.read_csv('F:/UdemyMachine2/blobs.csv')

X = df.copy()

outliers = LocalOutlierFactor(n_neighbors=20)
y_out = outliers.fit_predict(X)
df['lof'] = y_out
print(df.head(10))





# 80
# Zbadaj liczbę elementów odstających w zbiorze, tzn. zbadaj rozkład kolumny lof. Wynik wydrukuj do konsoli.
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


np.random.seed(42)

df = pd.read_csv('F:/UdemyMachine2/blobs.csv')

X = df.copy()

outliers = LocalOutlierFactor(n_neighbors=20)
y_out = outliers.fit_predict(X)
df['lof'] = y_out
print(df['lof'].value_counts())








