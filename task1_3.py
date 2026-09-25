"""Métricas de red a partir de la matriz de adyacencia (Task 1.3).

Implementa grado, coeficiente de clustering y distancia promedio usando
únicamente NumPy y la librería estándar. Al ejecutarse como script, verifica
las funciones con la matriz del Task 1.2.

Uso:
    python task1_3.py
"""

from collections import deque

import numpy as np


def grado(A):
    """Retorna el grado de cada nodo.

    Args:
        A: Matriz de adyacencia simétrica (N x N) de 0 y 1.

    Returns:
        Arreglo de tamaño N con el grado k_i de cada nodo.
    """
    return A.sum(axis=1)


def clustering(A):
    """Retorna el coeficiente de clustering de cada nodo.

    C_i = (aristas entre vecinos de i) / (k_i (k_i - 1) / 2).
    Los nodos con k_i < 2 tienen C_i = 0.

    Args:
        A: Matriz de adyacencia simétrica (N x N) de 0 y 1.

    Returns:
        Arreglo de tamaño N con el coeficiente C_i de cada nodo.
    """
    n = A.shape[0]
    k = grado(A)
    C = np.zeros(n)
    for i in range(n):
        if k[i] < 2:
            continue
        vecinos = np.flatnonzero(A[i])
        enlaces = A[np.ix_(vecinos, vecinos)].sum() / 2
        posibles = k[i] * (k[i] - 1) / 2
        C[i] = enlaces / posibles
    return C


def distancias_bfs(A, origen):
    """Retorna las distancias geodésicas desde un nodo mediante BFS.

    Args:
        A: Matriz de adyacencia simétrica (N x N) de 0 y 1.
        origen: Índice del nodo de partida.

    Returns:
        Arreglo de tamaño N con la distancia a cada nodo; -1 si no hay camino.
    """
    dist = np.full(A.shape[0], -1)
    dist[origen] = 0
    cola = deque([origen])
    while cola:
        u = cola.popleft()
        for v in np.flatnonzero(A[u]):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                cola.append(v)
    return dist


def distancia_promedio(A):
    """Retorna la distancia promedio de la red.

    Promedia las distancias geodésicas de todos los pares (i, j), i < j,
    que tienen camino entre sí.

    Args:
        A: Matriz de adyacencia simétrica (N x N) de 0 y 1.

    Returns:
        Distancia promedio <d>, o NaN si ningún par está conectado.
    """
    n = A.shape[0]
    total, pares = 0, 0
    for i in range(n):
        dist = distancias_bfs(A, i)[i + 1:]
        alcanzables = dist[dist > 0]
        total += alcanzables.sum()
        pares += alcanzables.size
    return total / pares if pares else float("nan")


def main():
    """Verifica las funciones con la matriz del Task 1.2."""
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ])

    esperado_k = np.array([2, 3, 3, 2, 2])
    esperado_C = np.array([1, 1 / 3, 1 / 3, 0, 0])
    esperado_d = 1.4

    k = grado(A)
    C = clustering(A)
    d = distancia_promedio(A)

    print(f"Grados:             {k}  -> coincide: {np.array_equal(k, esperado_k)}")
    print(f"Grado promedio:     {k.mean():.4f}  -> coincide: {np.isclose(k.mean(), 2.4)}")
    print(f"Clustering:         {np.round(C, 4)}  -> coincide: {np.allclose(C, esperado_C)}")
    print(f"Distancia promedio: {d:.4f}  -> coincide: {np.isclose(d, esperado_d)}")


if __name__ == "__main__":
    main()
