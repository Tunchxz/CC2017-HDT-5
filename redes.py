"""Utilidades para el Task 3: métricas de red, umbral epidémico y SIR en redes."""

import networkx as nx
import numpy as np
from scipy import stats


def umbral_epidemico(G):
    """Retorna el umbral crítico (beta/gamma)_c = <k> / (<k^2> - <k>).

    Args:
        G: Grafo no dirigido de NetworkX.

    Returns:
        Valor del umbral crítico.
    """
    k = np.array([d for _, d in G.degree()])
    return k.mean() / ((k**2).mean() - k.mean())


def metricas(G):
    """Calcula las métricas principales de una red.

    Si la red no es conexa, la distancia promedio se calcula sobre la
    componente conexa más grande.

    Args:
        G: Grafo no dirigido de NetworkX.

    Returns:
        Diccionario con <k>, <k^2>, C, <d> y el umbral (beta/gamma)_c.
    """
    k = np.array([d for _, d in G.degree()])
    gigante = G.subgraph(max(nx.connected_components(G), key=len))
    return {
        "<k>": k.mean(),
        "<k^2>": (k**2).mean(),
        "C": nx.average_clustering(G),
        "<d>": nx.average_shortest_path_length(gigante),
        "(β/γ)_c": umbral_epidemico(G),
    }


def simular_sir(G, beta, gamma, n_inicial, rng):
    """Simula una realización del modelo SIR en tiempo discreto sobre una red.

    En cada paso, cada infectado contagia a cada vecino susceptible con
    probabilidad `beta` y luego se recupera con probabilidad `gamma`. La
    simulación termina cuando no quedan infectados.

    Args:
        G: Grafo no dirigido de NetworkX con nodos 0..N-1.
        beta: Probabilidad de contagio por arista por paso.
        gamma: Probabilidad de recuperación por paso.
        n_inicial: Número de nodos infectados al inicio (elegidos al azar).
        rng: Generador `numpy.random.Generator`.

    Returns:
        Tupla (I_t, tamano_final): arreglo con I(t)/N en cada paso y la
        fracción de la población que llegó a infectarse (R final / N).
    """
    n = G.number_of_nodes()
    vecinos = [np.array(list(G.neighbors(i)), dtype=int) for i in range(n)]
    # 0 = S, 1 = I, 2 = R
    estado = np.zeros(n, dtype=int)
    estado[rng.choice(n, size=n_inicial, replace=False)] = 1

    I_t = [n_inicial / n]
    while (estado == 1).any():
        infectados = np.flatnonzero(estado == 1)
        nuevos = []
        for i in infectados:
            v = vecinos[i][estado[vecinos[i]] == 0]
            nuevos.append(v[rng.random(v.size) < beta])
        recuperados = infectados[rng.random(infectados.size) < gamma]
        estado[np.concatenate(nuevos)] = 1
        estado[recuperados] = 2
        I_t.append((estado == 1).sum() / n)

    return np.array(I_t), (estado == 2).sum() / n


def intervalo_confianza(x, nivel=0.95):
    """Retorna la media y el intervalo de confianza con la distribución t.

    Args:
        x: Arreglo de observaciones.
        nivel: Nivel de confianza (por defecto 0.95).

    Returns:
        Tupla (media, límite_inferior, límite_superior).
    """
    x = np.asarray(x)
    media = x.mean()
    margen = stats.t.ppf((1 + nivel) / 2, x.size - 1) * stats.sem(x)
    return media, media - margen, media + margen


def rellenar(trayectorias):
    """Iguala la longitud de las trayectorias rellenando con ceros al final.

    Args:
        trayectorias: Lista de arreglos 1D de distinta longitud.

    Returns:
        Matriz (n_trayectorias x longitud_máxima).
    """
    largo = max(t.size for t in trayectorias)
    return np.array([np.pad(t, (0, largo - t.size)) for t in trayectorias])
