
# In[1]:


import tensorflow  as tf
import tensorflow_datasets as tfds

#Descarga el set de datos de numeros escritos a mano MNIST, 70,000 numeros escritos a mano etiquetados
Datos, Metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

#Separar los datos del entrenamiento en dos variables, entrenamiento y pruebas
datos_ent, datos_test = Datos['train'], Datos['test']

#Funcion de normalizacion para los datos (Pasar valor de los pixeles de 0-255 a 0-1), asi la red aprende mejor y mas rapido
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 # Aqui se pasa de 0-255 a 0-1
    return imagenes, etiquetas

#Normalizar los datos de entrenamiento y los de prueba con la funcion anterior
datos_ent = datos_ent.map(normalizar)

datos_test = datos_test.map(normalizar)


# In[13]:


#Creamos el modelo (Modelo denso, regular, sin implementar redes convolusionales aun)
modelo = tf.keras.Sequential()

#Primera capa convolucional con matriz de 3x3
modelo.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu')) #1 canal porque es en blanco y negro
#Primera capa de agrupacion con matriz de 2x2
modelo.add(tf.keras.layers.MaxPooling2D(2,2))

#Segunda capa convolucional con matriz de 3x3
modelo.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
#Segunda capa de agrupacion con matriz de 2x2
modelo.add(tf.keras.layers.MaxPooling2D(2,2))

modelo.add(tf.keras.layers.Flatten())
modelo.add(tf.keras.layers.Dense(100, activation = 'relu'))
modelo.add(tf.keras.layers.Dense(10, activation = 'softmax'))


# In[14]:


#Compilamos el modelo
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.02),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


# In[16]:


#Los numeros de datos de entrenamiento y pruebas (60k y 10k)
num_datos_entrenamiento = Metadatos.splits["train"].num_examples
num_datos_pruebas = Metadatos.splits["test"].num_examples

#Trabajar por lotes
TAM_LOT=32

#Shuffle y repeat hacen que los datos esten mezclados de manera aleatoria
#para que el entrenamiento no se aprenda las cosas en orden
datos_entrenamiento = datos_ent.repeat().shuffle(num_datos_entrenamiento).batch(TAM_LOT)
datos_pruebas = datos_test.batch(TAM_LOT)


# In[19]:


#Realizar el entrenamiento
import math

historial = modelo.fit(
    datos_entrenamiento,
    epochs=60,
    steps_per_epoch=math.ceil(num_datos_entrenamiento/TAM_LOT)
)


# In[21]:


#Exportar el modelo para implementarlo al explorador
modelo.save('numeros_conv.h5')

