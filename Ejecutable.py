import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Carga de datos
# Especificamos la ruta al archivo CSV en nuestro sistema
ruta_csv = 'D:\\Datos_Abiertos\\proyecto\\datos_final.csv'

# Cargamos el archivo CSV
df = pd.read_csv(ruta_csv,encoding='utf-8' ,delimiter=';')

# Limpieza de datos
# Limpiamos el dataframe original, en este caso eliminamos filas con valores vacios en las columnas que se señalan
df_model=df.dropna(subset=['DOMICILIO_DIST'])
df_model=df_model.dropna(subset=['NACIMIENTO_DIST'])

# Eliminamos columnas que no se utilizaran para el modelo, asimismo convertimos las variables categóricas a numéricas utilizando one-hot encoding
df_encoded = df_model.drop(['IDHASH','id_temporal','COLEGIO_DEPA', 'COLEGIO_PROV','COLEGIO_DIST', 'ANIO', 'PERIODO', 'ANIO_PER','ANIO_PER_INGRESO','TIPO_MATRICULA', 'DOMICILIO_DEPA', 'DOMICILIO_PROV', 'DOMICILIO_DIST', 'ANIO_NACIMIENTO', 'NACIMIENTO_PAIS', 'NACIMIENTO_DEPA','NACIMIENTO_PROV', 'NACIMIENTO_DIST', 'Distancia (Colegio -  Domicilio)','METODOLOGIA', 'FACULTAD', 'CICLO_RELATIVOMAX' , 'N_MUESTRA'], axis=1)
df_encoded = pd.get_dummies(df_encoded, drop_first=False)

# Limpiamos el dataframe encoded, especificamente los nombres de las especialidades
correcciones_columnas = {
    'ESPECIALIDAD_CIENCIA DE LA COMPUTACIÃ“N': 'ESPECIALIDAD_CIENCIA DE LA COMPUTACIÓN',
    'ESPECIALIDAD_ESTADÃSTICA': 'ESPECIALIDAD_ESTADÍSTICA',
    'ESPECIALIDAD_FÃSICA': 'ESPECIALIDAD_FÍSICA',
    'ESPECIALIDAD_INGENIERÃA AMBIENTAL': 'ESPECIALIDAD_INGENIERÍA AMBIENTAL',
    'ESPECIALIDAD_INGENIERÃA CIVIL': 'ESPECIALIDAD_INGENIERÍA CIVIL',
    'ESPECIALIDAD_INGENIERÃA DE HIGIENE Y SEGURIDAD INDUSTRIAL': 'ESPECIALIDAD_INGENIERÍA DE HIGIENE Y SEGURIDAD INDUSTRIAL',
    'ESPECIALIDAD_INGENIERÃA DE MINAS': 'ESPECIALIDAD_INGENIERÍA DE MINAS',
    'ESPECIALIDAD_INGENIERÃA DE PETRÃ“LEO': 'ESPECIALIDAD_INGENIERÍA DE PETRÓLEO',
    'ESPECIALIDAD_INGENIERÃA DE PETRÃ“LEO Y GAS NATURAL': 'ESPECIALIDAD_INGENIERÍA DE PETRÓLEO Y GAS NATURAL',
    'ESPECIALIDAD_INGENIERÃA DE SISTEMAS': 'ESPECIALIDAD_INGENIERÍA DE SISTEMAS',
    'ESPECIALIDAD_INGENIERÃA DE SOFTWARE': 'ESPECIALIDAD_INGENIERÍA DE SOFTWARE',
    'ESPECIALIDAD_INGENIERÃA DE TELECOMUNICACIONES': 'ESPECIALIDAD_INGENIERÍA DE TELECOMUNICACIONES',
    'ESPECIALIDAD_INGENIERÃA ECONÃ“MICA': 'ESPECIALIDAD_INGENIERÍA ECONÓMICA',
    'ESPECIALIDAD_INGENIERÃA ELECTRÃ“NICA': 'ESPECIALIDAD_INGENIERÍA ELECTRÓNICA',
    'ESPECIALIDAD_INGENIERÃA ELÃ‰CTRICA': 'ESPECIALIDAD_INGENIERÍA ELÉCTRICA',
    'ESPECIALIDAD_INGENIERÃA ESTADÃSTICA': 'ESPECIALIDAD_INGENIERÍA ESTADÍSTICA',
    'ESPECIALIDAD_INGENIERÃA FÃSICA': 'ESPECIALIDAD_INGENIERÍA FÍSICA',
    'ESPECIALIDAD_INGENIERÃA GEOLÃ“GICA': 'ESPECIALIDAD_INGENIERÍA GEOLÓGICA',
    'ESPECIALIDAD_INGENIERÃA INDUSTRIAL': 'ESPECIALIDAD_INGENIERÍA INDUSTRIAL',
    'ESPECIALIDAD_INGENIERÃA MECATRÃ“NICA': 'ESPECIALIDAD_INGENIERÍA MECATRÓNICA',
    'ESPECIALIDAD_INGENIERÃA MECÃNICA': 'ESPECIALIDAD_INGENIERÍA MECÁNICA',
    'ESPECIALIDAD_INGENIERÃA MECÃNICA Y ELÃ‰CTRICA': 'ESPECIALIDAD_INGENIERÍA MECÁNICA Y ELÉCTRICA',
    'ESPECIALIDAD_INGENIERÃA METALÃšRGICA': 'ESPECIALIDAD_INGENIERÍA METALÚRGICA',
    'ESPECIALIDAD_INGENIERÃA NAVAL': 'ESPECIALIDAD_INGENIERÍA NAVAL',
    'ESPECIALIDAD_INGENIERÃA PETROQUÃMICA': 'ESPECIALIDAD_INGENIERÍA PETROQUÍMICA',
    'ESPECIALIDAD_INGENIERÃA QUÃMICA': 'ESPECIALIDAD_INGENIERÍA QUÍMICA',
    'ESPECIALIDAD_INGENIERÃA SANITARIA': 'ESPECIALIDAD_INGENIERÍA SANITARIA',
    'ESPECIALIDAD_INGENIERÃA TEXTIL': 'ESPECIALIDAD_INGENIERÍA TEXTIL',
    'ESPECIALIDAD_MATEMÃTICA': 'ESPECIALIDAD_MATEMÁTICA',
    'ESPECIALIDAD_QUÃMICA': 'ESPECIALIDAD_QUÍMICA'
}
# Aplicamos las correcciones a las columnas codificadas one-hot
df_encoded.columns = df_encoded.columns.to_series().replace(correcciones_columnas)

# Definición de características
# Definimos las características "X" y la variable objetivo "y"
X = df_encoded.drop(['Periodos_matriculados_total'], axis=1)
y = df_encoded['Periodos_matriculados_total']

# Escalamos las características con el fin de que se encuentren todas en la misma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construcción del Modelo de Red Neuronal
# Definimos la arquitectura de la red neuronal para la predicción de los periodos
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compilamos el modelo con el optimizador Adam y la función de pérdida de error cuadrático medio
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamos el modelo con datos de entrenamiento y lo validamos con datos de prueba
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

# Evaluación del Modelo
# Evaluamos el modelo en el conjunto de prueba
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Pérdida en el conjunto de prueba (MSE): {loss}')


# Guardado del Modelo
# Guardamos el modelo entrenado
model.save('D:/Datos_Abiertos/modelo_matriculas_red_neuronal.h5')

# Cargar el modelo
model = tf.keras.models.load_model('D:/Datos_Abiertos/modelo_matriculas_red_neuronal.h5')

# Evaluación y visualización del rendimiento del modelo en el conjunto de prueba
# Obtenenemos las predicciones del modelo
y_pred = model.predict(X_test)
y_pred = y_pred.flatten()

# Convertir y_test a unidimensional
y_test = np.array(y_test).flatten()

# Crear un DataFrame para comparar los valores reales y predichos
comparison_df = pd.DataFrame({
    'Valores Reales': y_test,
    'Valores Predichos': y_pred,
    'Diferencia': y_test - y_pred
})

# Mostrar las primeras filas del DataFrame
print(comparison_df.head())

# Crear un gráfico de dispersión para visualizar la comparación entre valores reales y predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Valores Predichos', alpha=0.6)
plt.scatter(y_test, y_test, color='green', label='Valores Reales', alpha=0.3)  # Opcional, para resaltar los valores reales
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Línea de Igualdad')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Comparación entre Valores Reales y Predichos')
plt.legend()
plt.grid(True)
plt.show()

# Ejemplo de aplicación
# Supongamos que tenemos un nuevo alumno con las siguientes características
nuevo_alumno = pd.DataFrame({
    'ANIOS_EN_UNI': [4],
    'MATRI_TRIKA_RIESGO': [0],
    'MATRI_N_TRIKA_RIESGO': [0],
    'MATRI_REINCORPORADO': [0],
    'MATRI_N_REINCORPORADO': [0],
    'EDAD_INGRESO': [20],
    'EDAD_ACTUAL': [23],
    'Distancia (UNI-Nacimiento)': [8007],
    'Distancia (Domicilio - UNI)': [8007],
    'SEXO': ['MASCULINO'],
    'MODALIDAD': ['ORDINARIO'],
    'ESPECIALIDAD': ['QUÍMICA'],
    'CICLO_RELATIVO': [7],
    'Peridos_matriculados_al_registro': [8]
})

# Convertimos variables categóricas a numéricas (utilizando one-hot encoding)
nuevo_alumno_encoded = pd.get_dummies(nuevo_alumno)

# Reindexar el DataFrame del nuevo alumno para asegurarse de que tenga las mismas columnas que las características "X"
nuevo_alumno_encoded = nuevo_alumno_encoded.reindex(columns=X.columns, fill_value=0)

# Escalamos las características del nuevo alumno
nuevo_alumno_scaled = scaler.transform(nuevo_alumno_encoded)

# Finalmente, predecimos
prediccion = model.predict(nuevo_alumno_scaled)
print(f'El número total de periodos en que el alumno se matriculará es: {prediccion[0][0]}')