# K-Means manual con el dataset de tracking de la NFL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df = pd.read_csv("input_2023_w01.csv")

print("*** Primeros registros del dataset ***")
print(df.head())
print("\n*** Informacion del dataset ***")
df.info()
print("\nTotal de registros:", len(df))

# solo me interesan velocidad y aceleracion
print("\n*** Estadisticas de s y a ***")
print(df[["s", "a"]].describe())



#el dataset es enorme, tomo 1 registro de cada 50
muestra = df[::50]
print("\nRegistros en la muestra:", len(muestra))

X = muestra[["s", "a"]]

#guardo los valores originales praa interpretar los clusters despues
info = muestra[["s", "a", "player_role", "player_position"]]


# escalar y separar train / test

# estandarizo porque K-Means usa distancias
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, info_train, info_test = train_test_split(X, info, test_size=0.2, random_state=1)
print("\nDatos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)



def distancia(p1, p2):
    # distancia euclidiana
    suma = 0
    for i in range(len(p1)):
        suma = suma + (p1[i] - p2[i]) ** 2
    return suma ** 0.5


def centroides_iniciales(datos, k):
    # los primeros k puntos ya estan revueltos por train_test_split, sirven como centroides al azar
    centroides = []
    for i in range(k):
        centroides.append(list(datos[i]))
    return centroides


def asignar_clusters(datos, centroides):
    # cada punto se va al centroide mas cercano
    asignaciones = []
    for punto in datos:
        cluster_cercano = 0
        distancia_minima = distancia(punto, centroides[0])
        for j in range(1, len(centroides)):
            d = distancia(punto, centroides[j])
            if d < distancia_minima:
                distancia_minima = d
                cluster_cercano = j
        asignaciones.append(cluster_cercano)
    return asignaciones


def calcular_centroides(datos, asignaciones, centroides):
    # el nuevo centroide es el promedio de los puntos del cluster
    k = len(centroides)
    dimensiones = len(datos[0])
    nuevos_centroides = []
    for j in range(k):
        suma = [0] * dimensiones
        cuenta = 0
        for i in range(len(datos)):
            if asignaciones[i] == j:
                for d in range(dimensiones):
                    suma[d] = suma[d] + datos[i][d]
                cuenta = cuenta + 1
        if cuenta > 0:
            promedio = []
            for d in range(dimensiones):
                promedio.append(suma[d] / cuenta)
            nuevos_centroides.append(promedio)
        else:
            # cluster vacio, se queda el centroide anterior
            nuevos_centroides.append(centroides[j])
    return nuevos_centroides


def inercia(datos, asignaciones, centroides):
    # suma de distancias al cuadrado de cada punto a su centroide
    total = 0
    for i in range(len(datos)):
        total = total + distancia(datos[i], centroides[asignaciones[i]]) ** 2
    return total


def kmeans(datos, k, max_iteraciones):
    centroides = centroides_iniciales(datos, k)
    historial_inercia = []
    iteracion = 0
    while True:
        asignaciones = asignar_clusters(datos, centroides)
        nuevos_centroides = calcular_centroides(datos, asignaciones, centroides)
        iteracion = iteracion + 1
        historial_inercia.append(inercia(datos, asignaciones, centroides))
        # paro cuando los centroides ya no cambian o se acaban las iteraciones
        if nuevos_centroides == centroides or iteracion == max_iteraciones:
            break
        centroides = nuevos_centroides
    asignaciones = asignar_clusters(datos, centroides)
    return centroides, asignaciones, historial_inercia


def indice_dunn(datos, asignaciones, centroides):
    # dunn = min(distancia entre clusters) / max(distancia dentro de cluster)
    k = len(centroides)
    minima_entre = -1
    for i in range(k):
        for j in range(i + 1, k):
            d = distancia(centroides[i], centroides[j])
            if minima_entre == -1 or d < minima_entre:
                minima_entre = d
    # distancia dentro del cluster = promedio de la distancia de sus puntos al centroide
    maxima_dentro = 0
    for j in range(k):
        suma = 0
        cuenta = 0
        for i in range(len(datos)):
            if asignaciones[i] == j:
                suma = suma + distancia(datos[i], centroides[j])
                cuenta = cuenta + 1
        if cuenta > 0:
            promedio = suma / cuenta
            if promedio > maxima_dentro:
                maxima_dentro = promedio
    return minima_entre / maxima_dentro


# metodo del codo para escogerK

max_iteraciones = 100

print("\n*** Metodo del codo ***")
valores_k = [1, 2, 3, 4, 5, 6, 7, 8]
inercias_codo = []
for k in valores_k:
    centroides, asignaciones, historial = kmeans(X_train, k, max_iteraciones)
    valor = inercia(X_train, asignaciones, centroides)
    inercias_codo.append(valor)
    print("K = %d   inercia = %.2f   iteraciones = %d" % (k, valor, len(historial)))

plt.figure()
plt.plot(valores_k, inercias_codo, marker="o")
plt.title("Metodo del codo")
plt.xlabel("K")
plt.ylabel("Inercia (WCSS)")


# entrenamiento

# con la grafica de codo se ve que despues de 3 la inercia ya casi no baja
k = 3

print("\n*** Entrenamiento con K =", k, "***")
centroides, asignaciones_train, historial = kmeans(X_train, k, max_iteraciones)

print("Iteraciones:", len(historial))
print("Inercia por iteracion:")
for i in range(len(historial)):
    print("  iteracion %d: %.2f" % (i + 1, historial[i]))

print("\nCentroides finales (datos escalados):")
for j in range(k):
    print("  Cluster %d: s = %.3f, a = %.3f" % (j, centroides[j][0], centroides[j][1]))

# promedio de s y a de cada clustr en unidades originales
asignaciones_train = np.asarray(asignaciones_train)
print("\nDescripcion de los clusters (entrenamiento):")
for j in range(k):
    puntos = info_train[asignaciones_train == j]
    print("  Cluster %d: %d registros, velocidad promedio = %.2f yd/s, aceleracion promedio = %.2f yd/s2"
          % (j, len(puntos), puntos["s"].mean(), puntos["a"].mean()))

plt.figure()
plt.plot(range(1, len(historial) + 1), historial, marker="o")
plt.title("Inercia por iteracion (K = %d)" % k)
plt.xlabel("Iteracion")
plt.ylabel("Inercia")



print("\n*** Prueba: asignacion de datos nuevos ***")
asignaciones_test = asignar_clusters(X_test, centroides)
asignaciones_test = np.asarray(asignaciones_test)

print("Registros de prueba por cluster:")
for j in range(k):
    print("  Cluster %d: %d registros" % (j, len(info_test[asignaciones_test == j])))

print("\nEjemplos de registros de prueba y el cluster asignado:")
for i in range(10):
    print("  s = %5.2f  a = %5.2f  rol = %-20s -> cluster %d"
          % (info_test["s"].iloc[i], info_test["a"].iloc[i], info_test["player_role"].iloc[i], asignaciones_test[i]))


# evaluacion

print("\n*** Evaluacion ***")
inercia_train = inercia(X_train, asignaciones_train, centroides)
inercia_test = inercia(X_test, asignaciones_test, centroides)
print("Inercia entrenamiento: %.2f  (promedio por punto: %.4f)" % (inercia_train, inercia_train / len(X_train)))
print("Inercia prueba:        %.2f  (promedio por punto: %.4f)" % (inercia_test, inercia_test / len(X_test)))
print("Indice de Dunn entrenamiento: %.4f" % indice_dunn(X_train, asignaciones_train, centroides))
print("Indice de Dunn prueba:        %.4f" % indice_dunn(X_test, asignaciones_test, centroides))


# graficas

# s contra a con un color por cluster, la X es el centroide (promedio del cluster)
plt.figure()
for j in range(k):
    puntos = info_train[asignaciones_train == j]
    plt.scatter(puntos["s"], puntos["a"], s=8, label="Cluster %d" % j)
    plt.scatter(puntos["s"].mean(), puntos["a"].mean(), marker="X", s=200, color="black")
plt.title("Clusters en datos de entrenamiento")
plt.xlabel("Velocidad s (yd/s)")
plt.ylabel("Aceleracion a (yd/s2)")
plt.legend()

plt.figure()
for j in range(k):
    puntos = info_test[asignaciones_test == j]
    plt.scatter(puntos["s"], puntos["a"], s=8, label="Cluster %d" % j)
    puntos_train = info_train[asignaciones_train == j]
    plt.scatter(puntos_train["s"].mean(), puntos_train["a"].mean(), marker="X", s=200, color="black")
plt.title("Asignacion de datos de prueba")
plt.xlabel("Velocidad s (yd/s)")
plt.ylabel("Aceleracion a (yd/s2)")
plt.legend()

plt.show()
