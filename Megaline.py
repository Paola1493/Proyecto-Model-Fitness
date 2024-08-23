#!/usr/bin/env python
# coding: utf-8

# # ¡Hola, Paola!  
# 
# Mi nombre es Carlos Ortiz, soy code reviewer de TripleTen y voy a revisar el proyecto que acabas de desarrollar.
# 
# Cuando vea un error la primera vez, lo señalaré. Deberás encontrarlo y arreglarlo. La intención es que te prepares para un espacio real de trabajo. En un trabajo, el líder de tu equipo hará lo mismo. Si no puedes solucionar el error, te daré más información en la próxima ocasión. 
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres**.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
#     
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# 
# Puedes responderme de esta forma: 
# 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# </div>
# ¡Empecemos!

# # ¿Cuál es la mejor tarifa?
# 
# Trabajas como analista para el operador de telecomunicaciones Megaline. La empresa ofrece a sus clientes dos tarifas de prepago, Surf y Ultimate. El departamento comercial quiere saber cuál de las tarifas genera más ingresos para poder ajustar el presupuesto de publicidad.
# 
# Vas a realizar un análisis preliminar de las tarifas basado en una selección de clientes relativamente pequeña. Tendrás los datos de 500 clientes de Megaline: quiénes son los clientes, de dónde son, qué tarifa usan, así como la cantidad de llamadas que hicieron y los mensajes de texto que enviaron en 2018. Tu trabajo es analizar el comportamiento de los clientes y determinar qué tarifa de prepago genera más ingresos.

# ## INTRODUCCIÓN
# 
# El propósito del proyecto es hacer un análisis estadístico de los ingresos que generan los planes de prepago de la empresa Megaline. Todo esto para poder hacer un ajuste en el presupuesto de publicidad.
# Para poder efectuar este análisis se deben realizar una serie de pasos para poder manipular de una manera mas eficiente el volumen de datos a trabajar. Es necesario visualizar, preparar, corregir y enriquezcer los datos para posteriormente proceder al análisis y poder responder la pregunta principal y las que vayan surgiendo en el camino. 
# 
# 
# 

# ## Inicialización

# In[1]:


# Cargar todas las librerías
import pandas as pd
import seaborn as sns 
import numpy as np 
from math import factorial
from scipy import stats as st
import math as mt
from matplotlib import pyplot as plt


# ## Cargar datos

# In[2]:


# Carga los archivos de datos en diferentes DataFrames

data_calls= pd.read_csv('/datasets/megaline_calls.csv')
data_internet= pd.read_csv('/datasets/megaline_internet.csv')
data_messages= pd.read_csv ('/datasets/megaline_messages.csv')
data_plans= pd.read_csv ('/datasets/megaline_plans.csv') 
data_users= pd.read_csv ('/datasets/megaline_users.csv') 


# In[ ]:





# ## Preparar los datos

# In[3]:


display(data_calls.info())
display(data_internet.info())
display(data_messages.info())
display(data_plans.info())
display(data_users.info())


# ## Tarifas

# In[4]:


# Imprime la información general/resumida sobre el DataFrame de las tarifas
display(data_plans.info())


# In[5]:


# Imprime una muestra de los datos para las tarifas

display(data_plans.head())


# COMENTARIO
# 
# Esta tabla está compuesta por 8 columnas, de los cuales 2 son datos float64, 5 int65 y 1 object. Las columnas están debidamente nombradas y no existen valores ausentes ni erróneos. Sin embargo, para poder efectuar futuras fusiones de data, realizaré un renombramiento de la columna 'plan_name' a 'plan' para que se titule de igual forma que en la tabla de usuarios.

# ## Corregir datos

# In[6]:


data_plans = data_plans.rename(columns={'plan_name': 'plan'})
display(data_plans.head())


# ## Enriquecer los datos

# In[ ]:





# ## Usuarios/as

# In[7]:


# Imprime la información general/resumida sobre el DataFrame de usuarios
display(data_users.info())


# In[8]:


# Imprime una muestra de datos para usuarios

display(data_users.head())
display(data_users.isna().sum())
display(data_users.duplicated().sum())


# COMENTARIO
# 
# En esta tabla se pueden observar 8 columnas, de las cuales 2 son datos float65 y 6 object.
# En la columna churn_date existen 466 datos ausentes que podrían ser reemplazados por 'active plan' a través del método fillna(). No hay datos duplicados ni erróneos

# ### Corregir los datos

# In[9]:


data_users['churn_date'].fillna('active plan', inplace=True) 
display(data_users.info())
display(data_users.head(30))


# ### Enriquecer los datos

# In[ ]:





# ## Llamadas

# In[10]:


# Imprime la información general/resumida sobre el DataFrame de las llamadas
display(data_calls.info())


# In[11]:


# Imprime una muestra de datos para las llamadas
display(data_calls.head())
display(data_calls.duplicated().sum())
data_calls['duration']= data_calls['duration'].apply(np.ceil)
display(data_calls.head())


# COMENTARIO
# 
# En esta tabla se pueden observar 4 columnas, de las cuales 1 son datos float65, 1 int64 y  2 object. Las columnas están debidamente nombradas y no existen valores ausentes, duplicados ni erróneos.

# ### Corregir los datos

# In[ ]:





# ### Enriquecer los datos

# In[12]:


data_calls['month'] = pd.DatetimeIndex(data_calls['call_date']).month
display(data_calls.head())
 


# COMENTARIO
# 
# En este punto procedí a extraer el mes en una columna aparte para poder trabajar de una manera más fácil los datos.

# ## Mensajes

# In[13]:


# Imprime la información general/resumida sobre el DataFrame de los mensajes
display(data_messages.info())


# In[14]:


# Imprime una muestra de datos para los mensajes
display(data_messages.head())
display(data_messages.duplicated().sum())


# COMENTARIO
# 
# En esta tabla se pueden observar 3 columnas, de las cuales 1 son datos int64 y 2 object. Las columnas están debidamente nombradas y no existen valores ausentes, duplicados ni erróneos.

# ### Corregir los datos

# In[ ]:





# ### Enriquecer los datos

# [Agrega factores adicionales a los datos si crees que pudieran ser útiles.]

# In[15]:


data_messages['month'] = pd.DatetimeIndex(data_messages['message_date']).month
display(data_messages.head())
 


# COMENTARIO
# 
# En esta tabla también extraje el mes en una columna aparte.

# ## Internet

# In[16]:


# Imprime la información general/resumida sobre el DataFrame de internet
display(data_internet.info())


# In[17]:


# Imprime una muestra de datos para el tráfico de internet
display(data_internet.head())
display(data_internet.duplicated().sum())


# COMENTARIO
# 
# En esta tabla se pueden observar 4 columnas, de las cuales 1 son datos float64, 1 int64 y 2 object. Las columnas están debidamente nombradas y no existen valores ausentes, duplicados ni erróneos.

# ### Corregir los datos

# In[ ]:





# ### Enriquecer los datos

# In[18]:


data_internet['month'] = pd.DatetimeIndex(data_internet['session_date']).month
display(data_internet.head())
 


# COMENTARIO
# 
# En esta tabla, al igual que en las otras, también extraje el mes en una columna aparte para poder trabajar de una manera más fácil los datos y posteriormente unir los DataFrames.

# ## Estudiar las condiciones de las tarifas

# In[19]:


# Imprime las condiciones de la tarifa y asegúrate de que te quedan claras
display(data_plans.head())


# In[20]:


# Calcula el número de llamadas hechas por cada usuario al mes. Guarda el resultado.
calls_for_user= data_calls.groupby(['user_id', 'month'])['id'].count().reset_index()
calls_for_user.head()


# COMENTARIO
# 
# En este punto usé el método groupby() para agrupar a los usuarios por meses y poder contabilizar, con el método count(), las llamadas que habían realizado.

# In[21]:


# Calcula la cantidad de minutos usados por cada usuario al mes. Guarda el resultado.
duration_calls= data_calls.groupby(['user_id', 'month'])['duration'].sum().reset_index()
display(duration_calls)


# COMENTARIO
# 
# Acá usé el método groupby() para agrupar a los usuarios por meses y, así poder sumar, con el método sum(), los minutos usados por cada usuario.

# In[22]:


# Calcula el número de mensajes enviados por cada usuario al mes. Guarda el resultado.
messages_for_user= data_messages.groupby(['user_id', 'month'])['id'].count().reset_index()
messages_for_user.head()


# COMENTARIO
# 
# 
# En este punto usé el método groupby() para agrupar a los usuarios por meses y poder contabilizar por 'id', con el método count(), los mensajes que habían enviado mensualmente.

# In[23]:


# Calcula el volumen del tráfico de Internet usado por cada usuario al mes. Guarda el resultado.
internet_for_user= data_internet.groupby(['user_id', 'month'])['mb_used'].sum().reset_index()
display(internet_for_user.head())
internet_for_user['gb_used']= internet_for_user['mb_used']/1024
internet_for_user['gb_used']= internet_for_user['gb_used'].apply(np.ceil)
display(internet_for_user.head())



# COMENTARIO
# 
# En este punto usé el método groupby() para agrupar a los usuarios por meses y, así poder sumar, con el método sum(), los mb usados por cada usuario.
# Posteriormente procedí a agregar otra columna en donde dejé el valor de los mb redondeados a gb para poder hacer cálculos futuros.

# In[24]:


# Fusiona los datos de llamadas, minutos, mensajes e Internet con base en user_id y month
df_1 = calls_for_user.merge(duration_calls, on=["user_id", "month"], how="outer").merge(messages_for_user, on=["user_id", "month"], how="outer").merge(internet_for_user, on=["user_id", "month"], how="outer").reset_index()
display(df_1)


# In[25]:


df_1 = df_1.rename(columns={'id_x': 'total_calls', 'id_y': 'total_messages'})
df_1.drop(['index'], axis=1, inplace= True )
display(df_1.head())


# COMENTARIO
# 
# En este punto, con el método merge(), fusioné las tablas de llamadas, mensajes e internet  y apliqué el método rename() para cambiarle el nombre a algunas columnas. Este DataFrame lo guardé como df_1

# In[26]:


# Añade la información de la tarifa
df_2=  data_users.merge(data_plans, on="plan", how= "outer")
display(df_2.head())


# COMENTARIO
# 
# En este apartado uní la tabla con los datos de los usuarios junto con la del detalle de los planes con el método merge(). Este DataFrame lo guardé como df_2.

# In[27]:


data_megaline= df_1.merge(df_2, on= 'user_id', how='outer')
display(data_megaline)


# In[28]:


data_megaline['total_calls'].fillna(0, inplace=True)
data_megaline['duration'].fillna(0, inplace=True)
data_megaline['total_messages'].fillna(0, inplace=True)
data_megaline['mb_used'].fillna(0, inplace=True)
data_megaline['gb_used'].fillna(0, inplace=True)
display(data_megaline)


# COMENTARIO
# 
# Uní el df_1 y df_2 mediante el método merge() para obtener una tabla que resumiera toda la información necesaria para proceder a realizar el análisis de los datos.

# In[29]:


# Calcula el ingreso mensual para cada usuario
data_megaline['min_charge'] = data_megaline['duration'] - data_megaline['minutes_included']
data_megaline['min_charge'] = data_megaline['min_charge'].apply(lambda x: max(x,0))

data_megaline['mess_charge'] = data_megaline['total_messages'] - data_megaline['messages_included']
data_megaline['mess_charge'] = data_megaline['mess_charge'].apply(lambda x: max(x,0))

data_megaline['gb_charge'] = data_megaline['mb_used'] - data_megaline['mb_per_month_included']
data_megaline['gb_charge'] = data_megaline['gb_charge'].apply(lambda x: max(x,0))
data_megaline['gb_rounded']= data_megaline['gb_charge']/1024
data_megaline['gb_rounded']= data_megaline['gb_rounded'].apply(np.ceil)


data_megaline['call_income'] = data_megaline['min_charge']*data_megaline['usd_per_minute']
data_megaline['messages_income'] = data_megaline['mess_charge']*data_megaline['usd_per_message']
data_megaline['internet_income'] = data_megaline['gb_rounded']*data_megaline['usd_per_gb']


data_megaline['total_income_per_user'] = data_megaline['call_income']+data_megaline['messages_income']+data_megaline['internet_income']+data_megaline['usd_monthly_pay'] 
display(data_megaline.head())    
    


# COMENTARIO
# 
# En este apartado agregué 8 columnas. En las primeras 3 calculé el cargo extra (x utilizados - x incluidos) por llamadas, mensajes y mb, éste último lo redondee y agregué en una 4ta columna para cálculos posteriores. En las siguiente 3 columnas agregué el ingreso por los cargos extras (el valor de la resta * usd_per_minute/message/gb) y en la última columna sumé los cargos extras más el valor del plan, el cual termina siendo el costo mensual total para cada usuario.

# ## Estudia el comportamiento de usuario

# ### Llamadas

# In[30]:


# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla.
calls_per_month= data_megaline[['duration', 'plan', 'month']]
calls_total= data_megaline.groupby(['month','plan' ])['duration'].mean().reset_index()

calls_surf= calls_total[(calls_total['plan'] == 'surf')][['month','plan','duration']].reset_index()

calls_ultimate= calls_total[(calls_total['plan'] == 'ultimate')][['month','plan','duration']].reset_index()

total_calls_per_month= calls_ultimate.merge(calls_surf, on= 'month', how='left')
total_calls_per_month.drop(['index_x', 'index_y'], axis=1, inplace= True )

total_calls_per_month.rename(columns={'plan_x': 'plan_ult', 'duration_x': 'duration_call_ult', 'plan_y': 'plan_surf', 'duration_y': 'duration_call_surf'}, inplace= True)
display(total_calls_per_month)

total_calls_per_month.plot(kind='bar', x= 'month', xlabel= 'Months of the year', ylabel='Avarage minutes', title= 'Avarage call minutes per month', figsize=(10,8))
plt.legend(['Ultimate plan', 'Surf plan'])
plt.show()


# COMENTARIO
# 
# 1. Lo primero que se puede observar en el gráfico es que ni el plan surf ni ultimate superaron los 500 minutos usados por los usuarios.
# 2. En general, la mayoría de los meses los dos planes tienen un comportamiento similar.

# In[31]:


# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.
sns.histplot(data = data_megaline, x = 'duration', hue = 'plan')


# In[32]:


# Calcula la media y la varianza de la duración mensual de llamadas.

variance_calls_surf= np.var(data_megaline[data_megaline['plan']== 'surf']['duration'])
variance_calls_ult= np.var(data_megaline[data_megaline['plan']== 'ultimate']['duration'])
mean_calls_surf= data_megaline[data_megaline['plan']== 'surf']['duration'].mean()
mean_calls_ult= data_megaline[data_megaline['plan']== 'ultimate']['duration'].mean()
print('La varianza de la duración de llamadas del plan surf es:', variance_calls_surf)
print('La varianza de la duración de llamadas del plan ultimate es:', variance_calls_ult)
print('La media de la duración de llamadas del plan surf es:', mean_calls_surf)
print('La media de la duración de llamadas del plan ultimate es:', mean_calls_ult)


# In[33]:


# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas
sns.boxplot(data = data_megaline, x = 'duration', y = 'plan')


# 
# 
# Según lo que se puede observar, el comportamiento de los clientes es similar entre los dos planes, tienen una distribución y una media cercana entre sí, la que podemos ver reflejada en el diagrama de caja. Sin embargo, lo que hace la diferencia es la cantidad de usuarios que hay en cada plan, siendo surf el que supera a ultimate en este punto, el cual podría ser el motivo por el que los datos estadísticos sean similares, esto se puede ver reflejado en el histograma. 

# ### Mensajes

# In[34]:


# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
messages_per_month= data_megaline[['total_messages', 'plan', 'month']]
messages_total= data_megaline.groupby(['month','plan' ])['total_messages'].mean().reset_index()

messages_surf= messages_total[(messages_total['plan'] == 'surf')][['month','plan','total_messages']].reset_index()

messages_ultimate= messages_total[(messages_total['plan'] == 'ultimate')][['month','plan','total_messages']].reset_index()

total_messages_per_month= messages_ultimate.merge(messages_surf, on= 'month', how='left')
total_messages_per_month.drop(['index_x', 'index_y'], axis=1, inplace= True )

total_messages_per_month.rename(columns={'plan_x': 'plan_ult', 'total_messages_x': 'total_messages_ult', 'plan_y': 'plan_surf', 'total_messages_y': 'total_messages_surf'}, inplace= True)
display(total_messages_per_month)

total_messages_per_month.plot(kind='bar', x= 'month', xlabel= 'Months of the year', ylabel='Messages per month', title= 'Avarage messages per month')
plt.legend(['Ultimate plan', 'Surf plan'])
plt.show()


# COMENTARIO
# 
# 1. Lo primero que se puede observar es que los clientes del plan ultimate envían más mensajes que los usuarios del plan surf.
# 2. El plan surf tiene una tendencia al crecimiento durante todo el año.

# In[35]:


# Compara el número de mensajes mensuales que necesitan los usuarios de cada plan. Traza un histograma.
sns.histplot(data = data_megaline, x = 'total_messages', hue = 'plan')


# In[36]:


# Calcula la media y la varianza de los mensajes.

variance_mess_surf= np.var(data_megaline[data_megaline['plan']== 'surf']['total_messages'])
variance_mess_ult= np.var(data_megaline[data_megaline['plan']== 'ultimate']['total_messages'])
mean_mess_surf= data_megaline[data_megaline['plan']== 'surf']['total_messages'].mean()
mean_mess_ult= data_megaline[data_megaline['plan']== 'ultimate']['total_messages'].mean()
print('La varianza de la cantidad de mensajes del plan surf es:',variance_mess_surf)
print('La varianza de la cantidad de mensajes del plan ultimate es:',variance_mess_ult)
print('La media de la cantidad de mensajes del plan surf es:',mean_mess_surf)
print('La media de la cantidad de mensajes del plan surf es:',mean_mess_ult)


# In[37]:


#sns.boxplot(total_messages_per_month['total_messages_surf']) 
sns.boxplot(data = data_megaline, x = 'total_messages', y = 'plan')


# 
# 
# En este apartado tenemos un comportamiento similar que en el de las llamadas, las varianzas y las medias son cercanas.
# Como se puede observar en el histograma el largo de las barra nos muestra que existe una mayor cantidad de clientes en el plan surf que en ultimate, sin embargo, los usuarios de este último son los que suelen usar el servicio de mensajería con mayor frecuencia.

# In[ ]:





# ### Internet

# In[38]:


# Compara la cantidad de tráfico de Internet consumido por usuarios por plan

gb_used_per_month= data_megaline[['gb_used', 'plan', 'month']]
gb_used_total= data_megaline.groupby(['month','plan' ])['gb_used'].mean().reset_index()

gb_used_surf= gb_used_total[(gb_used_total['plan'] == 'surf')][['month','plan','gb_used']].reset_index()

gb_used_ultimate= gb_used_total[(gb_used_total['plan'] == 'ultimate')][['month','plan','gb_used']].reset_index()

total_gb_used_per_month= gb_used_ultimate.merge(gb_used_surf, on= 'month', how='left')
total_gb_used_per_month.drop(['index_x', 'index_y'], axis=1, inplace= True )

total_gb_used_per_month.rename(columns={'plan_x': 'plan_ult', 'gb_used_x': 'gb_used_ult', 'plan_y': 'plan_surf', 'gb_used_y': 'gb_used_surf'}, inplace= True)
display(total_gb_used_per_month)

total_gb_used_per_month.plot(kind='bar', x= 'month', xlabel= 'month', ylabel='mb used per month', title= 'Gb used per month', figsize=(10,8))
plt.legend(['Ultimate plan', 'Surf plan'])
plt.show()


# COMENTARIO
# 
# 1. Según lo que muestra este gráfico los usuarios del plan ultimate usan mayor cantidad de gigas incluidos en el plan.
# 2. A partir del mes 6 hay un comportamiento similar en el gasto de gigas entre los dos planes.

# In[39]:


# Compara el tráfico de internet mensual que necesitan los usuarios de cada plan. Traza un histograma.
sns.histplot(data = data_megaline, x = 'gb_used', hue = 'plan')


# In[40]:


# Calcula la media y la varianza del tráfico de internet.

variance_internet_surf= np.var(data_megaline[data_megaline['plan']== 'surf']['gb_used'])
variance_internet_ult= np.var(data_megaline[data_megaline['plan']== 'ultimate']['gb_used'])
mean_internet_surf= data_megaline[data_megaline['plan']== 'surf']['gb_used'].mean()
mean_internet_ult= data_megaline[data_megaline['plan']== 'ultimate']['gb_used'].mean()
print('La varianza del tráfico de internet del plan surf es:', variance_internet_surf)
print('La varianza del tráfico de internet del plan ultimate es:',variance_internet_ult)
print('La media del tráfico de internet del plan ultimate es:', mean_internet_surf)
print('La media del tráfico de internet del plan ultimate es:', mean_internet_ult)


# In[41]:


#sns.boxplot(total_gb_used_per_month['gb_used_surf']) 
sns.boxplot(data = data_megaline, x = 'gb_used', y = 'plan')


# 
# 
# Al igual que en los casos anteriores podemos ver varianzas, medias y una distribución  muy parecidas. Sólo en los primeros 5 meses existe mayor diferencia en los gb utilizados. 

# ## Ingreso

# In[42]:


income_per_month= data_megaline[['total_income_per_user', 'plan', 'month']]
income_total= data_megaline.groupby(['month','plan' ])['total_income_per_user'].mean().reset_index()
income_per_surf= income_total[(income_total['plan'] == 'surf')][['month','plan','total_income_per_user']].reset_index()
income_per_ult= income_total[(income_total['plan'] == 'ultimate')][['month','plan','total_income_per_user']].reset_index()

total_income_per_month=income_per_ult.merge(income_per_surf, on= 'month', how='left')
total_income_per_month.drop(['index_x', 'index_y'], axis=1, inplace= True )
total_income_per_month.rename(columns={'plan_x': 'plan_ult', 'total_income_per_user_x': 'total_income_per_month_ult', 'plan_y': 'plan_surf', 'total_income_per_user_y': 'total_income_per_month_surf'}, inplace= True)
display(total_income_per_month)


total_income_per_month.plot(kind='bar', x= 'month', xlabel= 'month', ylabel='income per month', title= 'Income per month', figsize=(10,8))
plt.legend(['Ultimate plan', 'Surf plan'])
plt.show()



# In[43]:


sns.histplot(data = data_megaline, x = 'total_income_per_user', hue = 'plan')


# In[44]:


# Calcula la media y la varianza del ingreso mensual.
variance_income_surf= np.var(data_megaline[data_megaline['plan']== 'surf']['total_income_per_user'])
variance_income_ult= np.var(data_megaline[data_megaline['plan']== 'ultimate']['total_income_per_user'])
mean_income_surf= data_megaline[data_megaline['plan']== 'surf']['total_income_per_user'].mean()
mean_income_ult= data_megaline[data_megaline['plan']== 'ultimate']['total_income_per_user'].mean()
print('La varianza  ingreso mensual del plan surf es:', variance_income_surf)
print('La varianza  ingreso mensual del plan ultimate es:',variance_income_ult)
print('La media  ingreso mensual del plan surf es:',mean_income_surf)
print('La media  ingreso mensual del plan ultimate es:',mean_income_ult)


# In[45]:


#sns.boxplot(total_income_per_month['total_income_per_month_surf']) 
sns.boxplot(data = data_megaline, x = 'total_income_per_user', y = 'plan')


# 
# 
# En este punto se puede observar que las varianzas y las medias son diferentes. 
# Como se puede ver en el histograma, los clientes del plan surf tienden a tener un comportamiento más disperso. Estos suelen pagar montos variados debido a los diferentes recargos que se les aplica por concepto de llamadas, mensajes y gb. Mientras que, por otro lado, el ingreso de los usuarios del plan ultimate no presenta muchas variaciones, ya que, a ser un plan con más minutos, mensajes y gb no se les aplican estos recargos.  

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Muy buen trabajo con el análisis gráfico.
# </div>
# 

# ## Prueba las hipótesis estadísticas

# In[46]:


data_surf = data_megaline[data_megaline['plan']== 'surf']['total_income_per_user']
data_ultimate = data_megaline[data_megaline['plan']== 'ultimate']['total_income_per_user']


# In[47]:


# Prueba las hipótesis
#H0 No existe diferencia entre los ingresos promedio de los planes Ultimate y Surf.
#H1 Existe diferencia entre los ingresos promedio de los planes Ultimate y Surf.

alpha = 0.05 
results = st.ttest_ind(data_surf, data_ultimate) 

print('valor p: ', results.pvalue) 

if results.pvalue < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula") 




# COMENTARIO
# 
# En este punto filtré el DataFrame por plan e ingreso y luego se procedí a eliminar los valores ausentes que habían en la tabla para poder realizar la prueba de hipótesis.
# Utilicé la prueba de hipótesis ttest_ind, ya que, lo que se está analizando son dos muestras que son estadísticamente diferentes. 
# 
# El resultado de la prueba arroja que como el valor p es 1.724423314124219e-08 se debe rechazar la hipótesis nula, es decir, que si existe una diferencia entre los ingresos promedio de los planes Ultimate y Surf.

# In[48]:


data_ny= data_megaline[data_megaline['city']=='New York-Newark-Jersey City, NY-NJ-PA MSA']['total_income_per_user']
data_no_ny= data_megaline[data_megaline['city']!='New York-Newark-Jersey City, NY-NJ-PA MSA']['total_income_per_user']


# In[49]:


# Prueba las hipótesis
#H0 El ingreso promedio de los usuarios del área de NY-NJ es igual al de los usuarios de otras regiones.
#H1 El ingreso promedio de los usuarios del área de NY-NJ es diferente al de los usuarios de otras regiones.

alpha = 0.05 
results = st.ttest_ind(data_ny, data_no_ny) 

print('valor p: ', results.pvalue) 

if results.pvalue < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula") 



# COMENTARIO
# 
# En este punto también filtré el DataFrame pero esta vez por ciudad e ingreso y luego se procedí a eliminar los valores ausentes que habían en la tabla para poder realizar la prueba de hipótesis.
# Al igual que en el punto anterior utilicé la prueba de hipótesis ttest_ind, ya que, también se está analizando son dos muestras que son estadísticamente diferentes. 
# 
# El resultado de la prueba arroja que como el valor p es 0.049745282774076104 se debe rechazar la hipótesis nula, es decir, que si existe una diferencia entre los ingresos promedio de los los usuarios de NY-NJ y del resto de las regiones.

# ## Conclusión general
# 
# 
# 1. Los usuarios tienen un comportamiento similar en el uso de llamadas, mensajes y gb.
# 
# 2. A los usuarios del plan surf se les aplicaron más recargos que a los del plan ultimate por minutos, mensajes y gb.
# 
# 3. Los usuarios del plan surf superan en número a los del plan ultimate, lo que ayudó a nivelar las estadísticas.
# 
# 4. El ingreso promedio mensual del plan ultimate es más alto a pesar de que son menos usuarios que en el plan surf. Esto se debe a que, si bien, los clientes del plan ultimate no generaban muchos recargos, los que generaban se sumaban a la cuota de base y esta era mucho mayor que la del plan surf. 
# 
# 5. El ingreso promedio de NY-NJ es diferente al del resto de las regiones.
# 
# 6. Según las observaciones y el resultado del análisis, el enfoque publicitario debería tenerlo el plan Ultimate, puesto que el plan surf ya tiene el doble de usuarios y éstos generan más recargos pero su ingreso mensual es menor. En cambio, el plan Ultimate, si bien, genera mayores ingresos mensuales, se podría captar un mayor número de clientes y los ingresos aumentarían de manera significativa en comparación con el plan Surf.
# 

# <div class="alert alert-block alert-danger">
#     
# # Comentarios genrales
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Gran trabajo, Paola. Nos quedan algunos elementos por corregir antes de poder aprobar tu proyecto. Recuerda actualizar tus conclusiones si algo cambia.
# </div>
# 

# <div class="alert alert-block alert-success">
#     
# # Comentarios genrales
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Todo corregido. Has aprobado un nuevo proyecto. ¡Felicitaciones!
# </div>
