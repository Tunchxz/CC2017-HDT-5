# Bitácora de IA

Herramienta utilizada: Claude Code (modelo Opus 5.5).

## Task 1.3

### Prompt

```text
Implementa en un script de Python `task1_3.py` las métricas básicas de una red a partir de su
matriz de adyacencia. Usa únicamente NumPy y la librería estándar. No uses NetworkX ni ninguna otra
librería de grafos.

Estructura del código:

1. `grado(A)`: recibe la matriz de adyacencia como arreglo de NumPy y devuelve el grado de cada
   nodo con `A.sum(axis=1)`.
2. `clustering(A)`: devuelve el coeficiente de clustering de cada nodo. Para cada nodo obtén sus
   vecinos con `np.flatnonzero`, cuenta los enlaces entre ellos con `A[np.ix_(vecinos, vecinos)]`
   dividido entre 2 y divide entre los enlaces posibles k(k-1)/2. Si k < 2, C_i = 0.
3. `distancias_bfs(A, origen)`: implementa BFS con `collections.deque` y devuelve un arreglo con la
   distancia geodésica desde el origen a cada nodo, usando -1 para los nodos sin camino.
4. `distancia_promedio(A)`: corre `distancias_bfs` desde cada nodo y promedia las distancias de los
   pares i < j que sí tienen camino. Si ningún par está conectado, devuelve NaN.
5. `main()`: aplica las funciones a esta matriz y compara contra los valores esperados con
   `np.array_equal`, `np.allclose` y `np.isclose`, imprimiendo cada resultado y si coincide.

       A = [[0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]]

   Valores esperados: grados [2, 3, 3, 2, 2], grado promedio 2.4, clustering
   [1, 1/3, 1/3, 0, 0] y distancia promedio 1.4.

Documenta con docstrings según PEP 257 (Args y Returns) y ejecuta `main()` bajo
`if __name__ == "__main__":`.
```

### ¿Por qué funcionó?

Funcionó porque la restricción de librerías es fácil de verificar, basta revisar los imports para confirmar que solo aparecen NumPy y `collections`. Al prohibir NetworkX de forma explícita, el modelo no puede recurrir a `nx.clustering` ni a `nx.shortest_path_length`, y queda obligado a implementar el cálculo sobre la matriz.

También fija la arquitectura antes de escribir código. El BFS queda aislado en `distancias_bfs`, así que `distancia_promedio` solo recorre los nodos y promedia. El valor -1 como marca de "sin camino" resuelve el caso de redes no conexas sin lógica adicional, porque filtrar las distancias positivas deja fuera tanto los pares inalcanzables como la distancia de un nodo a sí mismo. Y pedir el conteo de enlaces con `np.ix_` evita recorrer los pares de vecinos con dos ciclos anidados.

Finalmente, la matriz y los valores esperados se entregan como dato dentro del prompt. Así la verificación es automática, el script imprime si cada resultado coincide o no, sin que haya que comparar a ojo con los cálculos manuales. El caso k < 2 se menciona de forma explícita porque es justo el que provoca una división entre cero en el clustering.

## Task 3

### Prompt

```text
Crea un entorno `.venv` con un `requirements.txt` (numpy, networkx, matplotlib, scipy, pandas,
jupyter, ipykernel) y desarrolla la simulación en dos archivos: un módulo `redes.py` con las
funciones y un Jupyter Notebook `task3.ipynb` que las use. pandas se usa solo para mostrar tablas y
matplotlib solo para graficar.

Módulo `redes.py`:

1. `umbral_epidemico(G)`: devuelve (β/γ)_c = <k> / (<k²> - <k>) a partir de los grados de G.
2. `metricas(G)`: devuelve un diccionario con <k>, <k²>, C (`nx.average_clustering`), <d>
   (`nx.average_shortest_path_length` sobre la componente conexa más grande) y el umbral.
3. `simular_sir(G, beta, gamma, n_inicial, rng)`: SIR en tiempo discreto con estados 0 = S, 1 = I y
   2 = R guardados en un arreglo de NumPy y listas de vecinos precalculadas. Elige los infectados
   iniciales con `rng.choice(..., replace=False)`. En cada paso, cada infectado contagia a cada
   vecino susceptible con probabilidad beta y luego se recupera con probabilidad gamma; termina
   cuando I = 0. Devuelve el arreglo I(t)/N y el tamaño final del brote R/N.
4. `intervalo_confianza(x, nivel=0.95)`: devuelve la media y los límites con la distribución t de
   Student (`scipy.stats.t.ppf` y `scipy.stats.sem`).
5. `rellenar(trayectorias)`: iguala el largo de las trayectorias rellenando con ceros al final y
   devuelve una matriz.

Notebook `task3.ipynb`:

1. Semilla 42 para todo: `np.random.default_rng(42)` para la simulación y `seed=42` en los
   generadores de NetworkX.
2. Tres redes: Erdős-Rényi (N=500, p=0.02), Barabási-Albert (N=500, m=5) y Watts-Strogatz (N=500,
   k=6, p=0.1), guardadas en un diccionario por nombre.
3. Una tabla en pandas con las métricas de `metricas(G)` para las tres redes.
4. Una figura de 1×3 con P(k) de cada red: barras para ER y WS, y puntos en escala log-log para BA.
5. SIR con β = 0.04, γ = 0.03, 5 infectados iniciales y 50 realizaciones por red, guardando las
   trayectorias y los tamaños finales.
6. Una tabla con la media, el IC del 95 %, la desviación estándar, β/γ y (β/γ)_c de cada red.
7. Una figura de 1×3 con `sharey=True` con las 50 trayectorias de I(t)/N en líneas delgadas con
   alpha bajo y la media en una línea negra más gruesa. Ejes rotulados y leyenda.

Documenta el módulo con docstrings según PEP 257.
```

### ¿Por qué funcionó?

Funcionó porque separa el cálculo de la presentación. Las funciones viven en `redes.py` y el notebook solo las llama, por eso queda corto y legible. Además, la misma `metricas` y la misma `simular_sir` se aplican a las tres redes sin duplicar código, porque las redes se guardan en un diccionario y se recorren con un solo ciclo.

También resuelve de antemano los problemas típicos de este tipo de simulación. Pasar `rng` como parámetro y fijar la semilla en NetworkX hace que las redes y las 50 realizaciones se repitan igual en cada ejecución. Pedir $\left< d \right>$ sobre la componente conexa más grande evita el error de `average_shortest_path_length` cuando la red aleatoria no es conexa. Y `rellenar` resuelve que cada realización termine en un paso distinto, sin eso no se podrían promediar las trayectorias para la línea de la media.

Finalmente, todos los parámetros se entregan como dato: $N$, $p$, $m$ y $k$ de cada red, $\beta$, $\gamma$, los 5 infectados iniciales y las 50 realizaciones. La regla de contagio se describe paso a paso (primero contagiar, luego recuperar, hasta que $I = 0$), lo que deja poco margen para interpretar el modelo de otra forma. Y el formato de cada figura (1×3, log-log solo para BA, media en línea gruesa) queda definido por completo, así que no hace falta corregir las gráficas después.
