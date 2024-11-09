#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importamos las librerias que se necesitan para el desarrollo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Importamos las librerias que vamos a necesitar en el modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


# In[2]:


#Leemos el archivo correspondiente al desarrollo 
df = pd.read_csv('alertas 1.csv', delimiter= ',') 


# In[3]:


# Obtener el número de registros (filas)
num_filas = df.shape[0]
print(f"El DataFrame tiene {num_filas} registros.")


# In[37]:


#Identificamos los tipos de campos
print(df.dtypes)


# In[5]:


# Detectar valores nulos
null_values = df.isnull().sum()
print("Valores nulos en cada columna:")
print(null_values)

# Detectar valores en blanco
blank_values = df.applymap(lambda x: isinstance(x, str) and x.strip() == "").sum()
print("\nValores en blanco en cada columna:")
print(blank_values)


# In[6]:


# Elimino los valores nulos de la base 
df = df.dropna()


# In[7]:


#Verificamos que se eliminen los valores na
null_values = df.isnull().sum()
print("Valores nulos en cada columna:")
print(null_values)


# In[8]:


# Reemplazar -1 por 1 en las columnas 'nacionalidad_riesgosa' y 'supera_perfil_trx'
df['nacionalidad_riesgosa'] = df['nacionalidad_riesgosa'].replace(-1, 1)
df['supera_perfil_trx'] = df['supera_perfil_trx'].replace(-1, 1)


# In[9]:


# Lista de columnas a convertir en Booleanos
columns_to_convert = [
    'con_antecedentes', 'prensa_negativa', 'investigado', 'recien_vinculado', 
    'importador', 'exportador', 'nacionalidad_riesgosa', 'supera_perfil_trx', 'actor_publico'
]

# Paso 1: Convertir a entero (asumiendo que los valores pueden ser 'SI'/'NO' o True/False)
# Se puede hacer una conversión condicional a 1 y 0, en caso de que haya valores de texto como 'SI'/'NO'.
df[columns_to_convert] = df[columns_to_convert].applymap(lambda x: 1 if x in ['SI', True, 1] else 0)

# Paso 2: Convertir a booleano (0 -> False, 1 -> True)
df[columns_to_convert] = df[columns_to_convert].astype(bool)

# Verificar el cambio
print(df[columns_to_convert].head())


# In[10]:


# Convertir 'id_alerta' y 'num_doc' a tipo 'str'
df['id_alerta'] = df['id_alerta'].astype(str)
df['num_doc'] = df['num_doc'].astype(str)
df['nombre'] = df['nombre'].astype(str)

# Verificar el cambio
print(df[['id_alerta', 'num_doc', 'nombre']].dtypes)


# In[11]:


# Lista de variables a convertir a categóricas
variables_a_convertir = [
    'tipo_cliente', 'region',
    'linea_negocio', 'decision'
]

# Convertir las columnas especificadas a tipo categórico
for var in variables_a_convertir:
    df[var] = df[var].astype('category')

# Verificar los tipos de las columnas después de la conversión
print(df.dtypes)


# In[12]:


# mostramos graficamente las variables categoricas
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))  # Aumentamos el tamaño de la figura si es necesario

# Lista de columnas categóricas
categorical_cols = ['tipo_cliente', 'linea_negocio', 'region'
                   ]

# Graficar las variables categóricas
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)  
    sns.countplot(x=col, data=df)
    plt.title(f'Conteo de {col}')
    plt.xticks(rotation=45)

plt.tight_layout()  # Ajuste de los gráficos para que no se solapen
plt.show()


# In[13]:


# Lista de las variables booleanas en el DataFrame
bool_cols = [
    'actor_publico', 'con_antecedentes', 'prensa_negativa', 'investigado', 
    'recien_vinculado', 'importador', 'exportador', 'nacionalidad_riesgosa', 
    'supera_perfil_trx'
]

# Crear una figura y ejes para los gráficos
plt.figure(figsize=(15, 10))

# Graficar cada una de las variables booleanas
for i, col in enumerate(bool_cols, 1):
    plt.subplot(3, 3, i)  # 3 filas, 3 columnas
    sns.countplot(x=df[col], hue=df[col], palette='Set2', legend=False)  
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')

# Ajustar los espacios entre los subgráficos
plt.tight_layout()
plt.show()


# In[14]:


# Graficar la variable objetivo 'decision'
plt.figure(figsize=(15, 10))  # Tamaño de la figura

sns.countplot(x='decision', data=df, hue='decision', palette='Set2', legend=False)

# Añadir título y etiquetas
plt.title('Distribución de la variable objetivo: "decision"')
plt.xlabel('Decision')
plt.ylabel('Frecuencia')

# Mostrar el gráfico
plt.show()


# In[16]:


#Eliminar columnas no relevantes 
df = df.drop(['id_alerta', 'num_doc', 'nombre'], axis=1)  # Eliminar columnas 


# In[17]:


#Preprocesamiento de los datos
# Convertir las columnas categóricas a numéricas usando LabelEncoder
le = LabelEncoder()

# Columnas categóricas para codificar
categorical_columns = ['region', 'tipo_cliente', 'linea_negocio']#, 'decision'

# Aplicamos LabelEncoder a las columnas categóricas
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Convertir las columnas booleanas a 0 y 1
boolean_columns = ['actor_publico', 'con_antecedentes', 'prensa_negativa', 'investigado', 
                   'recien_vinculado', 'importador', 'exportador', 'nacionalidad_riesgosa', 
                   'supera_perfil_trx']

for col in boolean_columns:
    df[col] = df[col].astype(int)

# Verificar los tipos de datos después de la conversión
print(df.dtypes)


# In[18]:


#Separar las variables predictoras (X) y la variable objetivo (y)
X = df.drop('decision', axis=1)  # Variables predictoras
y = df['decision']  # Variable objetivo


# In[19]:


#Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[20]:


#Crear y entrenar el modelo de clasificación
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# In[21]:


#Evaluar el modelo
# Realizar predicciones
y_pred = clf.predict(X_test)

# Calcular la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy * 100:.2f}%")

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de Confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Validación cruzada (para obtener una medida más robusta de rendimiento)
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"\nValidación cruzada - Media de Accuracy: {cv_scores.mean() * 100:.2f}%")


# In[22]:


#Generar resultados en un DataFrame
# Crear un DataFrame con las predicciones y los valores reales
results_df = pd.DataFrame({
    'predicted_decision': y_pred,
    'actual_decision': y_test
})
# Mostrar los primeros resultados
print("\nResultados de predicción:")
print(results_df.head())


# In[23]:


# Concatenar 'results_df' con las variables predictoras 'X_test'
final_results_df = pd.concat([X_test.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

# Mostrar los primeros resultados
print("\nResultados finales con las predicciones y las variables predictoras:")
print(final_results_df.head())


# In[24]:


# Lista de columnas a convertir
columns_to_convert = [
    'con_antecedentes', 'prensa_negativa', 'investigado', 'recien_vinculado', 
    'importador', 'exportador', 'nacionalidad_riesgosa', 'supera_perfil_trx', 'actor_publico'
]

# Paso 1: Convertir a entero (asumiendo que los valores pueden ser 'SI'/'NO' o True/False)
# Se puede hacer una conversión condicional a 1 y 0, en caso de que haya valores de texto como 'SI'/'NO'.
final_results_df[columns_to_convert] = final_results_df[columns_to_convert].applymap(lambda x: 1 if x in ['SI', True, 1] else 0)

# Paso 2: Convertir a booleano (0 -> False, 1 -> True)
final_results_df[columns_to_convert] = final_results_df[columns_to_convert].astype(bool)

# Verificar el cambio
print(final_results_df[columns_to_convert].head())


# In[25]:


#exportar el Dataframe
final_results_df.to_excel(r'C:\Users\omvelez\OneDrive - Grupo Bancolombia\Solicitudes_Gerencia\Solicitud_Alertas\Insumos\Prueba_Tecnica.xlsx', index=False)

