# 61
# Wykorzystując klasę KMeans (ustaw parametr random_state=42) z pakietu scikit-learn wyznacz listę wartości WCSS 
# (Within-Cluster Sum-of-Squared) dla liczby klastrów od 2 do 9 włącznie. 
# Wartości WCSS zaokrąglij do drugiego miejsca po przecinku. Listę wydrukuj do konsoli.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


np.random.seed(42)

df = pd.read_csv('clusters.csv')

wcss = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(round(kmeans.inertia_,2))
print(wcss)

 


# 62
# Wykorzystując klasę KMeans z pakietu scikit-learn wyznaczono listę wartości WCSS (Within-Cluster Sum-of-Squared) dla liczby klastrów od 2 do 9 włącznie. 
# Wykorzystując metodę łokcia (elbow method) dokonaj wyboru odpowiedniej liczby klastrów (najlepiej stwórz pomocniczy wykres WCSS). Wynik wydrukuj do konsoli.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.read_csv('clusters.csv')

wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(round(kmeans.inertia_, 2))
    
plt.plot(range(2,10), wcss)
print(3)





# 63
# Wykorzystując klasę AgglomerativeClustering z pakietu scikit-learn dokonaj podziału danych na dwa klastry. 
# Dokonaj predykcji na podstawie zbudowanego modelu i przypisz numer klastra do każdej próbki w obiekcie df (nadaj nazwę kolumny cluster). 
# Wyświetl dziesięć pierwszych wierszy obiektu df.
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


df = pd.read_csv('clusters.csv')
agglo = AgglomerativeClustering(n_clusters=2)

y__agglo = agglo.fit_predict(df)
df['cluster'] = y__agglo
print(df)






# 64
# Wykorzystując klasę AgglomerativeClustering z pakietu scikit-learn dokonaj podziału danych na dwa klastry (wykorzystaj metrykę Manhattan). 
# Dokonaj predykcji na podstawie zbudowanego modelu i przypisz numer klastra do każdej próbki w obiekcie df (nadaj nazwę kolumny cluster). 
# Wyświetl dziesięć pierwszych wierszy obiektu df.
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('clusters.csv')
agglo = AgglomerativeClustering(n_clusters=2, metric='manhattan', linkage='complete')

y__agglo = agglo.fit_predict(df)
df['cluster'] = y__agglo
print(df.head(10))





# 65
# Wykorzystując klasę DBSCAN z pakietu scikit-learn dokonaj podziału danych na klastry. Ustaw odpowiednio argumenty:
# eps=0.6
# min_samples=7
# Dokonaj predykcji na podstawie tak zbudowanego modelu i przypisz numer klastra do każdej próbki w obiekcie df (nadaj nazwę kolumny cluster). 
# Wyświetl dziesięć pierwszych wierszy obiektu df.
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


df = pd.read_csv('F:/UdemyMachine2/clusters.csv')

dbscan = DBSCAN(eps=0.6, min_samples=7)
y_db = dbscan.fit_predict(df)
df['cluster'] = y_db
print(df.head(10))






# 66 
# Wykorzystując klasę DBSCAN z pakietu scikit-learn dokonaj podziału danych na klastry. Ustaw odpowiednio parametry:
# eps=0.6
# min_samples=7
# Dokonaj predykcji na podstawie tak zbudowanego modelu i przypisz numer klastra do każdej próbki w obiekcie df (nadaj nazwę kolumny cluster). 
# Wyświetl rozkład częstości próbek w każdym klastrze.
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


df = pd.read_csv('F:/UdemyMachine2/clusters.csv')

dbscan = DBSCAN(eps=0.6, min_samples=7)
y_db = dbscan.fit_predict(df)
df['cluster'] = y_db
print(df['cluster'].value_counts())











# 67
# Wczytaj plik pca.csv do obiektu DataFrame df. Plik zawiera trzy zmienne objaśniające var1, var2, var3 oraz zmienną docelową class. 
# Następnie przypisz do zmiennej X kolumny: var1, var2, var3, zaś do zmiennej y kolumnę class.
# Wykorzystując klasę StandardScaler dokonaj standaryzacji zmiennych w obiekcie X. Wyświetl dziesięć pierwszych wierszy obiektu X.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.random.seed(42)


df = pd.read_csv('F:/UdemyMachine2/pca.csv')
X = df.iloc[:,:3]
y = df['class']

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print(X_std[:10])






# 68
# Zaimplementuj algorytm PCA wykorzystując tablicę X_std. Wynik ogranicz do dwóch głównych komponentów PCA i przypisz do zmiennej X_pca. 
# Wydrukuj dziesięć pierwszych wierszy obiektu X_pca.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.set_printoptions(
    precision=8, suppress=True, edgeitems=5, linewidth=200
)
np.random.seed(42)
df = pd.read_csv('F:/UdemyMachine2/pca.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
X_std = scaler.fit_transform(X)



eig_vals, eig_vecs = np.linalg.eig(np.cov(X_std, rowvar=False))


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(reverse=True)

W = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
X_pca = X_std.dot(W)
print(X_pca[:10])






# 69
# Zaimplementowano algorytm PCA wykorzystując tablicę X_std i przypisano do zmiennej X_pca. 
# Zbuduj obiekt DataFrame o nazwie df_pca wykorzystując tablicę X_pca oraz zmienną y tak jak pokazano poniżej.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=8, suppress=True, edgeitems=5, linewidth=200)
np.random.seed(42)
df = pd.read_csv('F:/UdemyMachine2/pca.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

eig_vals, eig_vecs = np.linalg.eig(np.cov(X_std, rowvar=False))
eig_pairs = [
    (np.abs(eig_vals[i]), eig_vecs[:, i])
    for i in range(len(eig_vals))
]
eig_pairs.sort(reverse=True)

W = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
X_pca = X_std.dot(W)


df_pca = pd.DataFrame(data=np.c_[X_pca, y], columns=['pca_1', 'pca_2', 'class'])
df_pca['pca_2'] = -df_pca['pca_2']
print(df_pca.head(10))






# 70
# Wykorzystując klasę PCA z pakietu scikit-learn dokonaj analizy PCA z dwoma komponentami na obiekcie X_std i przypisz do zmiennej df_pca. 
# Wydrukuj dziesięć pierwszych wierszy tego obiektu (dodaj także kolumnę class) tak jak pokazano poniżej.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


np.set_printoptions(
    precision=8, suppress=True, edgeitems=5, linewidth=200
)
np.random.seed(42)
df = pd.read_csv('F:/UdemyMachine2/pca.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=2)

vals = pca.fit_transform(X_std)

df_pca = pd.DataFrame(data=np.c_[vals,y], columns=['pca_1', 'pca_2', 'class'])

print(df_pca.head(10))







