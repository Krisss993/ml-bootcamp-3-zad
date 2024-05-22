
# 51
# Wykorzystując klasę RandomForestClassifier z pakietu scikit-learn zbuduj model klasyfikacji dla podanych danych 
# (ustaw argument random_state=42). Dokonaj trenowania modelu na zbiorze treningowym oraz oceny na zbiorze testowym.
# W odpowiedzi wydrukuj do konsoli dokładność modelu (do czterech miejsc po przecinku) na zbiorze testowym tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)


classifier = RandomForestClassifier(random_state=42)

classifier.fit(X_train, y_train)
s = classifier.score(X_test, y_test)
print(f'Accuracy: {s:.4f}')






# 52
# Wykorzystując klasę RandomForestClassifier z pakietu scikit-learn zbuduj model klasyfikacji dla podanych danych. 
# Wykorzystując metodę przeszukiwania siatki oraz klasę GridSearchCV (ustaw argumenty scoring='accuracy', cv=5) z
# najdź optymalne wartości parametrów criterion, max_depth oraz min_samples_leaf. Wartości parametrów przeszukaj z podanych poniżej:
#dla criterion -> ['gini', 'entropy']
#dla max_depth -> [6, 7]
#dla min_samples_leaf -> [4, 5]
#Dokonaj trenowania na zbiorze treningowym oraz oceny na zbiorze testowym.
# W odpowiedzi wydrukuj do konsoli najbardziej optymalne wartości parametrów criterion,  max_depth oraz min_samples_leaf.
import numpy as np
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


np.random.seed(42)
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

X_train, X_test, y_train, y_test = train_test_split(data, target)


classifier = RandomForestClassifier(random_state=42)
params = {'criterion': ['gini', 'entropy'], 'max_depth':[6, 7], 'min_samples_leaf':[4,5]}
grid = GridSearchCV(estimator= classifier, param_grid=params, n_jobs=-1, scoring='accuracy', cv=2)

grid.fit(X_train, y_train)
print(grid.best_params_)






# 53
# Dokonaj wektoryzacji dokumentów za pomocą klasy CountVectorizer z pakietu scikit-learn. 
# Wynik wyświetl w postaci obiektu DataFrame tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python'
]


vectorizer = CountVectorizer()
vectorizer.fit(documents)
data = vectorizer.fit_transform(documents).toarray()
cols = vectorizer.get_feature_names_out()

df = pd.DataFrame(data=data, columns=cols)
print(df)





# 54
# Dokonaj wektoryzacji dokumentów za pomocą klasy CountVectorizer z pakietu scikit-learn. 
# Użyj argumentu stop_words i ustaw jego wartość na 'english'. Wynik wyświetl w postaci obiektu DataFrame tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python'
]

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(documents)
data = vectorizer.transform(documents).toarray()
cols = vectorizer.get_feature_names_out()
df = pd.DataFrame(data=data, columns=cols)
print(df)





# 55
# Dokonaj wektoryzacji dokumentów za pomocą klasy CountVectorizer z pakietu scikit-learn. 
# Użyj argumentu stop_words i ustaw jego wartość na 'english'. Ustaw także odpowiedni argument, 
# który pozwoli wydobyć n-gramy: unigramy i bigramy. Wynik wyświetl w postaci obiektu DataFrame tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python',
    'programming language'
]


vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
vectorizer.fit(documents)
data = vectorizer.transform(documents).toarray()
cols = vectorizer.get_feature_names_out()
df = pd.DataFrame(data=data, columns=cols)
print(df)







# 56
# Dokonaj wektoryzacji dokumentów wykorzystując klasę TfidfVectorizer z pakietu scikit-learn. 
# Wynik wyświetl w postaci obiektu DataFrame tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)
documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python',
    'programming language'
]



tfidf = TfidfVectorizer()
data_transformed = tfidf.fit_transform(documents).toarray()
cols = tfidf.get_feature_names_out()


df = pd.DataFrame(data=data_transformed, columns=cols)
print(df)








# 57
# Dokonaj wektoryzacji dokumentów wykorzystując klasę TfidfVectorizer z pakietu scikit-learn. 
# Używając argumentu stop_words usuń z wektoryzacji dwa słowa: 'is' oraz 'in'.
# Wynik wyświetl w postaci obiektu DataFrame tak jak pokazano poniżej.
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)
documents = [
    'python is a programming language',
    'python is popular',
    'programming in python',
    'object-oriented programming in python',
    'programming language'
]


tfidf = TfidfVectorizer(stop_words=['in', 'is'])
data_transformed = tfidf.fit_transform(documents).toarray()
cols = tfidf.get_feature_names_out()
df = pd.DataFrame(data=data_transformed, columns=cols)
print(df)








# 58
# Wczytaj plik data.csv do obiektu DataFrame (plik zawiera dwie zmienne x1 oraz x2). 
# Następnie zaimplementuj algorytm K-średnich pozwalających rozdzielić podane dane na dwa klastry.
# Wyznacz centroid każdego klastra i wydrukuj jego współrzędne do konsoli. 
# Zaokrąglij wynik do trzech miejsc po przecinku każdej ze współrzędnych.
import numpy as np
from numpy.linalg import norm
import pandas as pd
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
np.random.seed(42)

df = pd.read_csv('data.csv')
df

x1_min = df['x1'].min()
x2_min = df['x2'].min()

x1_max = df['x1'].max()
x2_max = df['x2'].max()

plt.scatter(df['x1'],df['x2'])
plt.scatter([x1_min,x1_max],[x2_min,x2_max])

point_1 = [random.choice(np.arange(x1_min,x1_max, 0.1)), random.choice(np.arange(x1_min,x1_max, 0.1))]
point_2 = [random.choice(np.arange(x2_min,x2_max, 0.1)), random.choice(np.arange(x2_min,x2_max, 0.1))]


plt.scatter(df['x1'],df['x2'])
plt.scatter(point_1, point_2)



df.drop(columns='cluster', inplace=True)

point_1 = [random.choice(np.arange(x1_min,x1_max, 0.1)), random.choice(np.arange(x1_min,x1_max, 0.1))]
point_2 = [random.choice(np.arange(x2_min,x2_max, 0.1)), random.choice(np.arange(x2_min,x2_max, 0.1))]
for e in range(3):

    for i in range(len(df)):
        df.loc[i,'cluster'] = np.where(norm(np.array([df.loc[i,'x1'], df.loc[i,'x2']]) - np.array(point_1)) < norm(np.array([df.iloc[i,0], df.iloc[i,1]]) - np.array(point_2)), 1, 0)
    point_1 = [round(df[df['cluster'] == 0]['x1'].mean(),3), round(df[df['cluster'] == 0]['x2'].mean(),3)]
    point_2 = [round(df[df['cluster'] == 1]['x1'].mean(),3), round(df[df['cluster'] == 1]['x2'].mean(),3)]
    plt.scatter(df[df['cluster'] == 0]['x1'], df[df['cluster'] == 0]['x2'], c='b', label='Cluster 0')
    plt.scatter(df[df['cluster'] == 1]['x1'], df[df['cluster'] == 1]['x2'], c='r', label='Cluster 1')
    plt.scatter(point_1[0], point_1[1], marker='x', c='g', label='Centroid 0')
    plt.scatter(point_2[0], point_2[1], marker='x', c='y', label='Centroid 1')
    plt.legend()
    plt.title('Iteration {}'.format(e+1))
    plt.show()
print(point_1)
print(point_2)








point_1 = [random.choice(np.arange(x1_min,x1_max, 0.1)), random.choice(np.arange(x1_min,x1_max, 0.1))]
point_2 = [random.choice(np.arange(x2_min,x2_max, 0.1)), random.choice(np.arange(x2_min,x2_max, 0.1))]

# Iteracje algorytmu k-średnich
for e in range(10):
    # Obliczenie przyporządkowania do klastrów
    for i in range(len(df)):
        dist_point1 = norm(np.array([df.loc[i, 'x1'], df.loc[i, 'x2']]) - np.array(point_1))
        dist_point2 = norm(np.array([df.loc[i, 'x1'], df.loc[i, 'x2']]) - np.array(point_2))
        df.loc[i, 'cluster'] = np.where(dist_point1 < dist_point2, 1, 0)
    
    # Obliczenie nowych punktów środkowych klastrów
    point_1 = [df[df['cluster'] == 0]['x1'].mean(), df[df['cluster'] == 0]['x2'].mean()]
    point_2 = [df[df['cluster'] == 1]['x1'].mean(), df[df['cluster'] == 1]['x2'].mean()]
    
    # Wizualizacja wyników
    plt.scatter(df[df['cluster'] == 0]['x1'], df[df['cluster'] == 0]['x2'], c='b', label='Cluster 0')
    plt.scatter(df[df['cluster'] == 1]['x1'], df[df['cluster'] == 1]['x2'], c='r', label='Cluster 1')
    plt.scatter(point_1[0], point_1[1], marker='x', c='g', label='Centroid 0')
    plt.scatter(point_2[0], point_2[1], marker='x', c='y', label='Centroid 1')
    plt.legend()
    plt.title('Iteration {}'.format(e+1))
    plt.show()
print(point_1)
print(point_2)








# 59
# Wykorzystując klasę KMeans z pakietu scikit-learn dokonaj podziału danych na trzy klastry. Ustaw argumenty:
# max_iter=1000
# random_state=42
#W odpowiedzi wydrukuj współrzędne środka każdego klastra.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


np.random.seed(42)
df = pd.read_csv(r'clusters.csv')

kmeans = KMeans(max_iter=1000, random_state=42, n_clusters=3)
clusters = kmeans.fit_predict(df)
print(kmeans.cluster_centers_)






# 60
# Wykorzystując klasę KMeans z pakietu scikit-learn dokonano podziału danych na trzy klastry.
# Dokonaj predykcji na podstawie zbudowanego modelu kmeans i przypisz numer klastra do każdej próbki w obiekcie df (nadaj nazwę kolumny 'y_kmeans').
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


np.random.seed(42)
df = pd.read_csv('clusters.csv')

kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=42)
kmeans.fit(df)
y_kmeans = kmeans.predict(df)
df['y_kmeans'] = y_kmeans
print(df.head(10))






