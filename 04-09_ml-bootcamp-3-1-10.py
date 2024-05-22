
# 1
# Z poniższego słownika data utwórz obiekt DataFrame i przypisz do zmiennej df. 
# Następnie zapoznaj się z obiektem df i sprawdź liczbę braków danych dla wszystkich kolumn. 
# Podaj procent braków, wynik zaokrąglij do drugiego miejsca po przecinku i wydrukuj do konsoli tak jak pokazano poniżej.
import numpy as np
import pandas as pd


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)

round(df.isnull().sum() / len(df), 2)








# 2
# Wykorzystując pakiet do uczenia maszynowego scikit-learn oraz klasę SimpleImputer uzupełnij braki danych dla kolumny weight wartością średnią. 
# Zmiany przypisz na stałe do obiektu df. W odpowiedzi wydrukuj obiekt df do konsoli.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)

si = SimpleImputer(missing_values=np.nan)
df[['weight']] = si.fit_transform(df[['weight']],'mean')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['weight']] = imputer.fit_transform(df[['weight']])

print(df)







# 3
# Braki danych dla kolumny weight zastąpiono średnią wartością tej kolumny wykorzystując klasę SimpleImputer z pakietu scikit-learn. 
# Wyświetl wartość średnią wstawioną w miejsce braków dla tej kolumny (wykorzystaj instancję klasy SimpleImputer).
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['weight']] = imputer.fit_transform(df[['weight']])

print(imputer.statistics_[0])










# 4
# Wykorzystując pakiet do uczenia maszynowego scikit-learn oraz klasę SimpleImputer uzupełnij braki danych dla kolumny price stałą wartością 99.0. 
# Zmiany przypisz na stałe do obiektu df i wydrukuj ten obiekt do konsoli.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)

imputer = SimpleImputer(missing_values=np.nan, fill_value=99.0, strategy='constant')
df[['price']] = imputer.fit_transform(df[['price']])

print(df)




# 5
# Wykorzystując pakiet do uczenia maszynowego scikit-learn oraz klasę SimpleImputer uzupełnij braki danych dla kolumny size 
# najczęściej pojawiającym się elementem tej kolumny. Zmiany przypisz na stałe do obiektu df i wydrukuj ten obiekt df do konsoli.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[['size']] = imputer.fit_transform(df[['size']])

print(df)






# 6
# Wytnij wszystkie wiersze obiektu df dla których kolumna weight nie przyjmuje wartość np.nan. 
# Na tak otrzymanym obiekcie policz wartość średnią dla kolumn numerycznych price oraz weight. Wynik wydrukuj do konsoli.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)

w = df.dropna(subset='weight')
print(w.mean(numeric_only=True))

w = df[~df['weight'].isnull()]
print(w.mean(numeric_only=True))





# 7
# Wydobądź z obiektu df kolumny typu object. Następnie uzupełnij wszystkie braki dla tych kolumn wartością 'empty'. 
# Przypisz wynik do zmiennej df_object i wydrukuj tą zmienną do konsoli.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)

df.info()
df.select_dtypes(include='object').fillna('empty')





# 8
# Dokonaj dyskretyzacji kolumny weight na 3 przedziały o równej szerokości. Wynik przypisz do nowej kolumny o nazwie 'weight_cut' tak jak pokazano poniżej. 
# W odpowiedzi wydrukuj obiekt df do konsoli.
# 75.0  (67.977, 75.667]
import pandas as pd


df = pd.DataFrame(data={'weight': [75., 78.5, 85., 91., 84.5, 83., 68.]})

df['weight_cut'] = pd.cut(df['weight'],bins=3)
print(df)






# 9
# Dokonaj dyskretyzacji kolumny weight na 3 przedziały o zadanej postaci:
# (60, 75]
# (75, 80]
# (80, 95]
# Wynik przypisz do nowej kolumny o nazwie 'weight_cut' tak jak pokazano poniżej. W odpowiedzi wydrukuj obiekt df do konsoli.
import pandas as pd


df = pd.DataFrame(data={'weight': [75., 78.5, 85., 91., 84.5, 83., 68.]})

df['weight_cut'] = pd.cut(df['weight'], bins= [60, 75, 80, 95])
print(df)




# 10
# Dokonaj dyskretyzacji kolumny weight na 3 przedziały o zadanej postaci:
# (60, 75]
# (75, 80]
# (80, 95]
# Wynik przypisz do nowej kolumny o nazwie 'weight_cut' tak jak pokazano poniżej. W odpowiedzi wydrukuj obiekt df do konsoli.
# oraz przypisz im odpowiednio etykiety: light, normal, heavy
import pandas as pd


df = pd.DataFrame(data={'weight': [75., 78.5, 85., 91., 84.5, 83., 68.]})

df['weight_cut'] = pd.cut(df['weight'], bins=[60, 75, 80, 95], labels=['light', 'normal', 'heavy'])
print(df)



