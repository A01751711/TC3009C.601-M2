# Clustering jerarquico aglomerativo con scikit-learn y scipy, dataset de tracking de la NFL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage



df = pd.read_csv("input_2023_w01.csv")

print("*** Primeros registros del dataset ***")
print(df.head())
print("\n*** Informacion del dataset ***")
df.info()
print("\nTotal de registros:", len(df))

# igual que en el proyecto anterior, solo uso velocidad y aceleracion
print("\n*** Estadisticas de s y a ***")
print(df[["s", "a"]].describe())



# el clustering jerarquico guarda una matriz de distancias entre todos los puntos,asi que con el dataset completo no cabe en memoria, tomo 1 registro de cada 50
muestra = df[::50]
print("\nRegistros en la muestra:", len(muestra))

X = muestra[["s", "a"]]

# guardo los valores originales para interpretar los clusters despues
info = muestra[["s", "a", "player_role", "player_position"]]


# escalar y separar train test

# estandarizo porque el algoritmo usa distancias
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, info_train, info_test = train_test_split(X, info, test_size=0.2, random_state=1)
print("\nDatos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)



#funciones de apoyo (no son parte del algoritmo, son para evaluar y para asignar datos nuevos)

def calcular_centroides(datos, etiquetas, k):
    # el centroide de cada cluster es el promedio de sus puntos
    centroides = []
    for j in range(k):
        centroides.append(datos[etiquetas == j].mean(axis=0))
    return np.array(centroides)


def asignar_clusters(datos, centroides):
    # el clustering jerarquico no tiene predict, cada dato nuevo se manda al centroide mas cercano
    asignaciones = []
    for punto in datos:
        distancias = np.sqrt(((centroides - punto) ** 2).sum(axis=1))
        asignaciones.append(distancias.argmin())
    return np.array(asignaciones)


def inercia(datos, etiquetas, centroides):
    # suma de distancias al cuadrado de cada punto a su centroide
    total = 0
    for i in range(len(datos)):
        total = total + ((datos[i] - centroides[etiquetas[i]]) ** 2).sum()
    return total


def indice_dunn(datos, etiquetas, centroides):
    # dunn = min(distancia entre centroides) / max(distancia promedio dentro de un cluster)
    k = len(centroides)
    minima_entre = -1
    for i in range(k):
        for j in range(i + 1, k):
            d = np.sqrt(((centroides[i] - centroides[j]) ** 2).sum())
            if minima_entre == -1 or d < minima_entre:
                minima_entre = d
    maxima_dentro = 0
    for j in range(k):
        puntos = datos[etiquetas == j]
        promedio = np.sqrt(((puntos - centroides[j]) ** 2).sum(axis=1)).mean()
        if promedio > maxima_dentro:
            maxima_dentro = promedio
    return minima_entre / maxima_dentro


def plot_dendrogram(model, **kwargs):
    # arma la matriz de linkage que necesita scipy a partir del modelo de sklearn (funcion vista en clase)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)



# comparacion de tipos de linkage

# pruebo los linkages que vimos en clase con K = 3 y me quedo con el de mejor silhouette
print("\n*** Comparacion de linkage (K = 3) ***")
tipos_linkage = ["ward", "complete", "average", "single"]
for tipo in tipos_linkage:
    modelo = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage=tipo)
    etiquetas = modelo.fit_predict(X_train)
    tamanos = np.bincount(etiquetas)
    print("%-9s silhouette = %.4f   tamanos = %s" % (tipo, silhouette_score(X_train, etiquetas), tamanos))



# dendrograma para escoger K

# distance_threshold=0 y n_clusters=None hacen que se construya el arbol completo
modelo_completo = AgglomerativeClustering(distance_threshold=0, n_clusters=None, metric="euclidean", linkage="ward")
modelo_completo = modelo_completo.fit(X_train)

# las ultimas uniones son las de mayor distancia, ahi se ve el salto para escoger el corte
print("\n*** Distancias de las ultimas uniones del dendrograma (ward) ***")
ultimas = modelo_completo.distances_[-8:]
for i in range(len(ultimas)):
    print("  union %d: distancia = %.2f  -> %d clusters" % (i + 1, ultimas[i], len(ultimas) - i))

plt.figure(figsize=(9, 5))
plt.title("Dendrograma (ward), ultimos 5 niveles")
plot_dendrogram(modelo_completo, truncate_mode="level", p=5)
plt.xlabel("Numero de puntos en el nodo (o indice del punto si no hay parentesis)")
plt.ylabel("Distancia")



# silhouette para varios K

print("\n*** Silhouette por numero de clusters (ward) ***")
valores_k = [2, 3, 4, 5, 6, 7, 8]
silhouettes = []
for k in valores_k:
    modelo = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
    etiquetas = modelo.fit_predict(X_train)
    valor = silhouette_score(X_train, etiquetas)
    silhouettes.append(valor)
    print("K = %d   silhouette = %.4f" % (k, valor))

plt.figure()
plt.plot(valores_k, silhouettes, marker="o")
plt.title("Silhouette por numero de clusters")
plt.xlabel("K")
plt.ylabel("Silhouette")



# entrenamiento

# el dendrograma y el silhouette apuntan a K = 3
k = 3

print("\n*** Entrenamiento con K =", k, "***")
modelo = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
etiquetas_train = modelo.fit_predict(X_train)

print("Parametros del modelo:", modelo.get_params())
print("Numero de uniones realizadas:", len(modelo.children_))

# centroides de cada cluster, se usan para asignar datos nuevos y para las metricas
centroides = calcular_centroides(X_train, etiquetas_train, k)

print("\nCentroides (datos escalados):")
for j in range(k):
    print("  Cluster %d: s = %.3f, a = %.3f" % (j, centroides[j][0], centroides[j][1]))

print("\nDescripcion de los clusters (entrenamiento):")
for j in range(k):
    puntos = info_train[etiquetas_train == j]
    print("  Cluster %d: %d registros, velocidad promedio = %.2f yd/s, aceleracion promedio = %.2f yd/s2"
          % (j, len(puntos), puntos["s"].mean(), puntos["a"].mean()))



# prueba: asignacion de datos nuevos

print("\n*** Prueba: asignacion de datos nuevos ***")
etiquetas_test = asignar_clusters(X_test, centroides)

print("Registros de prueba por cluster:")
for j in range(k):
    print("  Cluster %d: %d registros" % (j, len(info_test[etiquetas_test == j])))

print("\nEjemplos de registros de prueba y el cluster asignado:")
for i in range(10):
    print("  s = %5.2f  a = %5.2f  rol = %-20s -> cluster %d"
          % (info_test["s"].iloc[i], info_test["a"].iloc[i], info_test["player_role"].iloc[i], etiquetas_test[i]))



# evaluacion

print("\n*** Evaluacion ***")
inercia_train = inercia(X_train, etiquetas_train, centroides)
inercia_test = inercia(X_test, etiquetas_test, centroides)
print("Silhouette entrenamiento: %.4f" % silhouette_score(X_train, etiquetas_train))
print("Silhouette prueba:        %.4f" % silhouette_score(X_test, etiquetas_test))
print("Inercia entrenamiento: %.2f  (promedio por punto: %.4f)" % (inercia_train, inercia_train / len(X_train)))
print("Inercia prueba:        %.2f  (promedio por punto: %.4f)" % (inercia_test, inercia_test / len(X_test)))
print("Indice de Dunn entrenamiento: %.4f" % indice_dunn(X_train, etiquetas_train, centroides))
print("Indice de Dunn prueba:        %.4f" % indice_dunn(X_test, etiquetas_test, centroides))


# matriz de confusion contra k-Means

# no hay etiqueta verdadera, asi que comparo contra K-Means de sklearn con el mismo k y los mismos datos
print("\n*** Comparacion con K-Means ***")
kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
kmeans_train = kmeans.fit_predict(X_train)
kmeans_test = kmeans.predict(X_test)

# los numeros de cluster de cada metodo no coinciden, cada cluster jerarquico se empareja con el cluster de K-Means con el que mas registros comparte
tabla = confusion_matrix(etiquetas_train, kmeans_train)
print("Cruce original (filas = jerarquico, columnas = K-Means):")
print(tabla)
mapeo = tabla.argmax(axis=1)
print("Mapeo de clusters jerarquico -> K-Means:", mapeo)

kmeans_train_mapeado = np.zeros(len(kmeans_train), dtype=int)
kmeans_test_mapeado = np.zeros(len(kmeans_test), dtype=int)
for j in range(k):
    kmeans_train_mapeado[kmeans_train == mapeo[j]] = j
    kmeans_test_mapeado[kmeans_test == mapeo[j]] = j

matriz_train = confusion_matrix(etiquetas_train, kmeans_train_mapeado)
matriz_test = confusion_matrix(etiquetas_test, kmeans_test_mapeado)
print("\nMatriz de confusion entrenamiento (filas = jerarquico, columnas = K-Means):")
print(matriz_train)
print("Concordancia entrenamiento: %.4f" % (np.trace(matriz_train) / matriz_train.sum()))
print("\nMatriz de confusion prueba (filas = jerarquico, columnas = K-Means):")
print(matriz_test)
print("Concordancia prueba: %.4f" % (np.trace(matriz_test) / matriz_test.sum()))



# graficas

plt.figure()
for j in range(k):
    puntos = info_train[etiquetas_train == j]
    plt.scatter(puntos["s"], puntos["a"], s=8, label="Cluster %d" % j)
    plt.scatter(puntos["s"].mean(), puntos["a"].mean(), marker="X", s=200, color="black")
plt.title("Clusters jerarquicos en datos de entrenamiento")
plt.xlabel("Velocidad s (yd/s)")
plt.ylabel("Aceleracion a (yd/s2)")
plt.legend()

plt.figure()
for j in range(k):
    puntos = info_test[etiquetas_test == j]
    plt.scatter(puntos["s"], puntos["a"], s=8, label="Cluster %d" % j)
    puntos_train = info_train[etiquetas_train == j]
    plt.scatter(puntos_train["s"].mean(), puntos_train["a"].mean(), marker="X", s=200, color="black")
plt.title("Asignacion de datos de prueba")
plt.xlabel("Velocidad s (yd/s)")
plt.ylabel("Aceleracion a (yd/s2)")
plt.legend()

# comparacion visual con K-Means en entrenamiento
plt.figure()
for j in range(k):
    puntos = info_train[kmeans_train_mapeado == j]
    plt.scatter(puntos["s"], puntos["a"], s=8, label="Cluster %d" % j)
plt.title("K-Means (sklearn) en datos de entrenamiento")
plt.xlabel("Velocidad s (yd/s)")
plt.ylabel("Aceleracion a (yd/s2)")
plt.legend()

plt.show()
