#!/usr/bin/env python
# coding: utf-8

# # Proyecto 13

# Hola Paola!
# 
# Soy **Patricio Requena** 👋. Es un placer ser el revisor de tu proyecto el día de hoy!
# 
# Revisaré tu proyecto detenidamente con el objetivo de ayudarte a mejorar y perfeccionar tus habilidades. Durante mi revisión, identificaré áreas donde puedas hacer mejoras en tu código, señalando específicamente qué y cómo podrías ajustar para optimizar el rendimiento y la claridad de tu proyecto. Además, es importante para mí destacar los aspectos que has manejado excepcionalmente bien. Reconocer tus fortalezas te ayudará a entender qué técnicas y métodos están funcionando a tu favor y cómo puedes aplicarlos en futuras tareas. 
# 
# _**Recuerda que al final de este notebook encontrarás un comentario general de mi parte**_, empecemos!
# 
# Encontrarás mis comentarios dentro de cajas verdes, amarillas o rojas, ⚠️ **por favor, no muevas, modifiques o borres mis comentarios** ⚠️:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>

# Con el fin de combatir la cancelación, Model Fitness ha digitalizado varios de sus perfiles de clientes. Tu tarea consiste en analizarlos y elaborar una estrategia de retención de clientes.
# 
# Tienes que:
# 
# Aprender a predecir la probabilidad de pérdida (para el próximo mes) para cada cliente.
# 
# Elaborar retratos de usuarios típicos: selecciona los grupos más destacados y describe sus características principales.
# 
# Analizar los factores que más impactan la pérdida.
# 
# Sacar conclusiones básicas y elaborar recomendaciones para mejorar la atención al cliente:
# 
# identificar a los grupos objetivo;
# 
# sugerir medidas para reducir la rotación;
# 
# describir cualquier otro patrón que observes con respecto a la interacción con los clientes.

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Buen trabajo incluyendo esta sección para explicar el proyecto, te recomendaría que agregues una sección más donde hagas un breve resumen de lo que trata el proyecto y su objetivo a parte de lo que ya se detalló
# </div>

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch


# <div class="alert alert-block alert-info">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Una buena práctica para cuando tengas que importar varias librerías es seguir el siguiente órden en las mismas:
# 
# - Primero todas las librerías que vienen ya con python cómo `datetime`, `os`, `json`, etc.
# - Luego de las librerías de Python si las de terceros cómo `pandas`, `scipy`, `numpy`, etc.
# - Por último, en el caso de que armes tu propio módulo en tu proyecto esto debería ir en tercer lugar, y recuerda siempre ordenar cada tipo por orden alfabético
# </div>

# ## Análisis exploratorio de datos (EDA)

# In[2]:


# Cargar el dataset
df = pd.read_csv('/datasets/gym_churn_us.csv')

# Observación de características ausentes
missing_values = df.isnull().sum()
print("Valores ausentes por característica:\n", missing_values)


# In[3]:


df.head()


# In[4]:


# Resumen estadístico
df.describe()


# In[5]:


# Valores medios por grupo
grupo_0 = df[df['Churn'] == 0].mean()
grupo_1 = df[df['Churn'] == 1].mean()
display(grupo_0)
display(grupo_1)


# In[6]:


# Crear histogramas de barras y distribuciones por grupo
fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(15, 20))

# Ajustar en caso de que no sea divisible por 3
if len(df.columns) % 3 != 0:
    fig, axes = plt.subplots(len(df.columns)//3 + 1, 3, figsize=(15, 20))

for i, column in enumerate(df.columns):
    if column != 'cancelacion':
        sns.histplot(df[df['Churn'] == 1][column], kde=True, color='red', ax=axes[i//3, i%3], label='Cancelacion')
        sns.histplot(df[df['Churn'] == 0][column], kde=True, color='blue', ax=axes[i//3, i%3], label='No Cancelacion')
        axes[i//3, i%3].set_title(column)
        axes[i//3, i%3].legend()

plt.tight_layout()
plt.show()


# In[7]:


# Matriz de correlación
correlation_matrix = df.corr()


# In[8]:


# Mostrar la matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Correcto! la visualización es muy clara para poder comparar
# </div>

# ## Construir un modelo para predecir la cancelación de usuarios

# In[9]:


# Definir las características (X) y el objetivo (y)
X = df.drop(columns='Churn') 
y = df['Churn']

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Mostrar el tamaño de los conjuntos de entrenamiento y validación
print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de validación:", X_val.shape)


# In[10]:


# Entrenar el modelo de regresión logística
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[11]:


# Predicciones del conjunto de validación con regresión logística
y_pred_log_reg = log_reg.predict(X_val)

# Evaluar el modelo de regresión logística
log_reg_accuracy = accuracy_score(y_val, y_pred_log_reg)
log_reg_precision = precision_score(y_val, y_pred_log_reg)
log_reg_recall = recall_score(y_val, y_pred_log_reg)

print("Regresión Logística:")
print("Exactitud:", log_reg_accuracy)
print("Precisión:", log_reg_precision)
print("Recall:", log_reg_recall)


# In[12]:


# Entrenar el modelo de bosque aleatorio
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Predicciones del conjunto de validación con bosque aleatorio
y_pred_rf = rf.predict(X_val)

# Evaluar el modelo de bosque aleatorio
rf_accuracy = accuracy_score(y_val, y_pred_rf)
rf_precision = precision_score(y_val, y_pred_rf)
rf_recall = recall_score(y_val, y_pred_rf)

print("Bosque Aleatorio:")
print("Exactitud:", rf_accuracy)
print("Precisión:", rf_precision)
print("Recall:", rf_recall)


# In[13]:


# Comparación de los modelos
print("\nComparación de Modelos:")
print("Regresión Logística: Exactitud =", log_reg_accuracy, ", Precisión =", log_reg_precision, ", Recall =", log_reg_recall)
print("Bosque Aleatorio: Exactitud =", rf_accuracy, ", Precisión =", rf_precision, ", Recall =", rf_recall)


# El modelo de regresión logística tiene un mejor desempeño en todas las métricas comparadas (exactitud, precisión y recall) con el bosque aleatorio. Por lo tanto, parece ser el mejor modelo para este conjunto de datos específico.

# ## Crear clústeres de usuarios/as

# In[14]:


# Dejar de lado la columna 'churn'
X = df.drop(columns='Churn')

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear la matriz de distancias
linked = sch.linkage(X_scaled, method='ward')


# In[15]:


# Trazar el dendrograma
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linked)
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia Euclídea')
plt.show()


# In[16]:


# Entrenar el modelo K-means
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X_scaled)

# Añadir los clústeres al dataframe original
df['cluster'] = clusters

# Valores medios de las características para cada clúster
cluster_means = df.groupby('cluster').mean()
print("Valores medios de las características para cada clúster:\n", cluster_means)



# In[17]:


# Trazar distribuciones de características para cada clúster
for column in X.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue='cluster', multiple='stack', palette='Set1')
    plt.title(f'Distribución de {column} por Clúster')
    plt.show()


# 1. La distribución de géneros es mimilar.
# 2. Existe una gran diferencia entre clientes que viven cerca comparado con los que viven lejos.
# 3. En cuanto a 'Partner'existen variaciones significativas entre clústers, tomando en cuenta los clientes que son partes de empresas asociadas y clientes que no.
# 4. Por otro lado, 'Promo_friends' presenta gran diferencia entre clientes que llegaron por la oferta y los que no.
# 5. La mayoría de los clústers tienen clientes que contratan el servicio por menos de 2 meses y muchos del clúster 1 por 12 meses.
# 6. La edad de la mayoría de los clientes por clúster se encuentra entre los 25 y 35 años.
# 7. Los cargos adicionales sen encuentran entre 0 y 200, distribuyéndose de manera similar entre clústers.
# 8. El promedio de visitas por semana está entre 1 y 3 días, y se distribuye de forma similar entre clústers en ese rango.
# 9. Lo mismo pasa con la distribución de visitas mensual, sin embargo, el promedio de días varía entre 1 y 4 días.

# In[18]:


# Calcular la tasa de cancelación para cada clúster
cancellation_rate = df.groupby('cluster')['Churn'].mean()*100
print("Tasa de cancelación para cada clúster:\n", cancellation_rate)


# Cluster 0: Tasa de cancelación del 26.6%
# 
# Cluster 1: Tasa de cancelación del 2.7%
# 
# Cluster 2: Tasa de cancelación del 52.1%
# 
# Cluster 3: Tasa de cancelación del 44.1%
# 
# Cluster 4: Tasa de cancelación del 7.2%
# 
# Identificación de clústeres propensos a irse y leales
# 
# Más propensos a irse: 
# 
# Cluster 2: 52.1%
# 
# Cluster 3: 44.1%
# 
# Más leales:
# 
# Cluster 1: 2.7%
# 
# Cluster 4: 7.2%
# 
# El Cluster 0 tiene una tasa de cancelación intermedia del 26.6%.

# ## Conclusiones
# 
# Estrategias de Retención para Clústeres con Altas Tasas de Cancelación:
# 
# 1. Realizar encuestas y entrevistas para identificar problemas específicos que enfrentan los usuarios.
# 2. Implementar programas de fidelización personalizados, como descuentos exclusivos o servicios adicionales.
# 3. Mejorar las características de productos o servicios que están insatisfaciendo a estos usuarios, basándose en el feedback recibido.
# 
# Implementación de marketing:
# 
# Campañas de Retención:
# 
# 1. Emails personalizados: Enviar correos electrónicos con ofertas exclusivas y encuestas de satisfacción a los usuarios.
# 
# 2. Atención al cliente: Ofrecer soporte adicional condtante, abordando problemas antes de que presenten mayores inconvenientes.
# 
# Campañas de Fidelización:
# 
# 1. Programa de recompensas: Crear un programa de recompensas basado en puntos para usuarios, donde puedan canjear puntos por productos o servicios.
# 
# 2. Eventos: Invitar a estos usuarios a eventos exclusivos, como lanzamientos de nuevos productos o sesiones de feedback.
# 
# Mejoras de Producto/Servicio:
# 
# 1. Actualizaciones basadas en feedback: Utilizar la retroalimentación de usuarios leales para implementar nuevas mejoras que también podrían atraer y retener a usuarios de otros clústeres.
# 
# 2. Pruebas A/B: Realizar pruebas A/B con diferentes mejoras en características específicas para ver cuál tiene el mayor impacto en la satisfacción y retención de usuarios.
# 
# 

# <div class="alert alert-block alert-info">
# <b>Comentario general (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Desarrollaste un proyecto muy bueno y lograste obtener los resultados esperados acorde a las instrucciones dadas, además tus recomendaciones al final son muy buenas y se nota la relación que tienen con el proceso de análisis y entrenamiento de los modelos que realizaste con anterioridad.
#     
# Veo que incluso incluiste en ciertos puntos tu interpretación de las gráficas mostradas, te recomiendo que apliques esto para el resto de gráficas ya que algunos quedaron sin comentarios. El poner tu interpretación agiliza el entender el proceso de análisis cuando alguien más en tu equipo quiera usar tu notebook.
#     
# Saludos!
# </div>
