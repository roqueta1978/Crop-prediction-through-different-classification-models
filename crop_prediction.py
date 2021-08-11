'''Modelo de clasificación para estimar el cultivo más adecuado en función de diferentes variables agronómicas'''

# Librerías
## Funciones algebráicas
import numpy as np
## Tratamiento de datos
import pandas as pd
## Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
## Preprocesamiento de datos
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
## Separación de datos
from sklearn.model_selection import train_test_split
## Modelos de Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
## Métricas de precisión de los Modelos
from sklearn.metrics import accuracy_score
## Eliminación de los avisos
import warnings
warnings.filterwarnings('ignore')

''' Información del set de datos:

    La agricultura de precisión está de moda en la actualidad. Ayuda a los agricultores
    a tomar decisiones informadas sobre la estrategia agrícola. Aquí se muestra
    un conjunto de datos que podría permitir construir un modelo predictivo
    para recomendar, en base a diferentes parámetros, que resultaría más adecuado
    cultivar en una finca en particular.

    Contexto
    El conjunto de datos se corresponde con registros de precipitaciones, clima
    y fertilizantes disponibles para la India.

    Campos de información
    · N - relación de contenido de nitrógeno en el suelo
    · P - proporción de contenido de fósforo en el suelo
    · K - proporción de contenido de potasio en el suelo
    · temperature - temperatura en grados Celsius
    · humidity - humedad relativa en%
    · ph - valor de ph del suelo
    · rainfall - precipitación en mm
'''
# Carga del set de datos

df=pd.read_csv('crop_recommendation.csv')
df.head()
## Comprobación de la presencia de valores nulos

### Información del set de datos
df.info()

print(df.isnull().sum())

plt.figure(figsize=(9,9))
sns.heatmap(df.isnull(), cbar=False, cmap='Set1')
plt.show()
'''Como se puede observar, no existe presencia de valores nulos'''

# Análisis exploratorio de los datos (EDA):

'''En base a la información del set de datos, se puede establecer que 7 variables
   son numéricas, 3 podrían ser discretas (N, P, K) y 4 continuas
   (temperature, humidity, ph y rainfall). Por último, hay una variable nominal
   que se corresponde con el tipo de cultivo adecuado según esas condiciones
   establecidas.
'''
## Estadística descriptiva

df.describe()
'''Como se puede observar a través de las medidas de dispersión, las variables numéricas podrían presentar
   distribuciones no acordes a la normalidad y la presencia de valores atípicos.'''

### Distribución de las variables numéricas

num_var=df.select_dtypes(include=['int64', 'float64']).columns
num_var

plt.figure(figsize=(15,15))
for i, col in enumerate(num_var):
    plt.subplot(3,3,i+1)
    sns.histplot(df[col], stat='count', kde=True, color='peru')
    plt.xlabel(col, weight='bold')
    plt.ylabel('Count', weight='bold')
plt.tight_layout(pad=1.1)
plt.show()

'''Solo las variables temperatura y ph parecen mostrar una distribución que
   podría asemejarse a la de una normal
   '''
### Presencia de outliers

plt.figure(figsize=(15,15))
for i, col in enumerate(num_var):
    plt.subplot(3,3,i+1)
    sns.boxplot(df[col], color='peru')
    plt.xlabel(col, weight='bold')
    plt.ylabel(None)
plt.tight_layout(pad=1.1)
plt.show()

''' Se observa la presencia de valores atípicos en prácticamente todas
    las variables numéricas'''

## Relación entre las variables numéricas:

### Matríz de correlación

plt.figure(figsize=(12,9))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='rainbow')
plt.title('Matriz de Correlación')
plt.show()

'''Como se puede ver, solo parece existir una elevada correlación entre la variable
   fósforo (P) y potasio (K).'''

## Relación entre la variable objeto y las variables en estudio.
num_var
plt.figure(figsize=(15,15))
for i, col in enumerate(num_var):
    df_group=df.groupby(by=['label']).agg({'N':'mean',
                                           'P':'mean',
                                           'K':'mean',
                                           'temperature':'mean',
                                           'humidity':'mean',
                                           'ph':'mean',
                                           'rainfall':'mean'}) \
                                           .reset_index() \
                                           .sort_values(by=[col], ascending=False)
    plt.subplot(3,3,i+1)
    sns.barplot(x=col, y='label', data=df_group, palette='Reds_r')
    plt.ylabel('Crop', weight='bold')
    plt.xlabel(col, weight='bold')
plt.tight_layout(pad=1.1)
plt.show()

'''Como se puede observar para las distintas variables en estudio, existe variabilidad dependiendo del cultivo. Este hecho hace
   que resulte muy posible poder establecer un modelo de clasificación eficiente.
   Para profundizar un poco más en este hecho, se pueden seleccionar aletoriamente 5 cultivos y observar que tipo de distribución muestra
   para cada una de las variables en estudio. Así, se puede ver con mayor claridad la variabilidad observada entre cultivos.
'''

## Variabilidad en 5 cultivos seleccionados de forma aleatoria.

sample=np.array(df_group['label'].sample(n=5, random_state=42))

sample


df_sample=df[df['label']=='rice']
for crop in sample[1:]:
    df_sample=df_sample.append(df[df['label']==crop])
df_sample

### Distribución de las variables estudio para cada uno de los cultivos

#### Histograma
plt.figure(figsize=(15,15))
for i, col in enumerate(num_var):
    plt.subplot(3,3,i+1)
    sns.histplot(data=df_sample, x=col, hue='label', stat='count',
                 kde=True, palette=['red', 'green', 'blue', 'yellow', 'cyan'])
    plt.xlabel(col, weight='bold')
    plt.ylabel('count', weight='bold')
plt.tight_layout(pad=1.1)
plt.show()

#### Boxplot

plt.figure(figsize=(15,15))
for i, col in enumerate(num_var):
    plt.subplot(3,3,i+1)
    sns.boxplot(data=df_sample, x='label', y=col,
                palette=['red', 'green', 'blue', 'yellow', 'cyan'])
    plt.xlabel('crop', weight='bold')
    plt.ylabel(col, weight='bold')
plt.tight_layout(pad=1.1)
plt.show()


'''Como se puede observar a través de los gráficos de distribución, algunos de los cultivos muestran diferencias claras con respecto
   a las variables en estudio. Esto permite constatar que existiendo variabilidad podría ser posible establecer eficientemente un modelo
   de clasficación. No obstante, no existe mejor manera de constatarlo que implementar un modelo clasificación sobre los datos.
'''


# Implementación del modelo de clasificación.
'''Se van a implementar distintos modelos de clasificación se va a estimar cual es el más preciso'''

## Preprocesamiento de las variables.
'''Se va a realizar un preprocesado de la variables para transformar los datos y que puedan así
ser interpretados de la mejor manera posible durante la implementación de los modelos.'''
### Variables numéricas

df
'''Como se puede ver, las variables numéricas muestran distintos órdenes de magnitud.
   Por ello, será necesario llevar a cabo un procesado, el cual permita que todas presenten
   un orden magnitud similar y así evitar que unas tengan mayor peso que otras.

   El método de transformación se corresponde con una estandarización, donde la media será 0 y
   la desviación estándar será 1.
'''

scale=StandardScaler()

for col in num_var:
    df[col]=scale.fit_transform(df[[col]])

df.head()

### Variable objeto

'''La variable objeto se corresponde con una variable nominal representada por el cultivo.
   Para que sea interpretado correctamente por los algoritmos de clasificación, conviene transformar
   esta variable a un código numérico. Como en este caso no existe ningún tipo de orden establecido,
   se aplicará el método LabelEncoder().
'''

le=LabelEncoder()
df['label']=le.fit_transform(df[['label']])

df.head()

#### Comprobación del número de etiquetas

df['label'].unique()
'''Hay 22 etiquetas, equivalentes a los 22 cultivos que se encuentran en estudio'''

## Separación de datos

X=df.loc[:, df.columns!='label']

y=df['label']


'''Se va a utilizar durante la aplicación del método train_test_split
   el parámetro stratify para asegurar que entrenamiento y evaluación
   exista la misma proporción de cada una de las clases
'''


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

'''Se puede observar que existe la misma proporción de clases en entrenamiento que
   en evaluación
'''

## Modelado

model={'Decision Tree':DecisionTreeClassifier(),
       'Random Forest':RandomForestClassifier(),
       'K_neighbors':KNeighborsClassifier(),
       'Gradient Boosting':GradientBoostingClassifier(),
       'MLP':MLPClassifier(),
       'Support Vector':SVC()}


results={}
results['model']=[]
results['acc_train']=[]
results['acc_test']=[]

for key, clf in model.items():
    clf.fit(X_train, y_train)
    y_pred_train=clf.predict(X_train)
    y_pred_test=clf.predict(X_test)
    results['model'].append(key)
    results['acc_train'].append(accuracy_score(y_train, y_pred_train))
    results['acc_test'].append(accuracy_score(y_test, y_pred_test))


results

df_result=pd.DataFrame(results)
df_result

'''Como se puede observar, los modelos de clasificación seleccionados
   tienen una enorme capacidad de predecir correctamente las clases.
   Además, tal y como se puede ver, la precisión es muy alta en entrenamiento y
   evaluación, lo que significa que no existe overfitting y generalizan bien
   posteriormente con datos no conocidos.

   Con los resultados obtenidos, es muy difícil seleccionar que modelo resulta
   más preciso, pues todos son perfectamente válidos. No obstante, si se tuviera
   que seleccionar alguno en concreto, posiblemente estaría entre
   el RandomForestClassifier, el GradientBoostingClassifier o el MLPClassifier.

   Por otro lado, resulta importante volver a enfatizar en el hecho de que los
   las variables estudiadas, tal y como se ha observado en el "EDA", aportan
   una información precisa que permite establecer la variabilidad existente entre
   cultivos para las condiciones estudiadas. Seguramente, este factor es el que
   resulta más determinante para que los modelos de clasificación hayan resultado
   tan precisos.
'''
