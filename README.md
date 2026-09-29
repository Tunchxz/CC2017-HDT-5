# Hoja de Trabajo 5

Esta actividad analiza la estructura y la dinámica de redes reales y sintéticas:

- **Task 1.3** (`task1_3.py`): calcula el grado, el coeficiente de clustering y la distancia promedio (con BFS) usando solo NumPy, y verifica los resultados con la matriz del Task 1.2.
- **Task 3** (`task3.ipynb`): genera redes Erdős-Rényi, Barabási-Albert y Watts-Strogatz, compara sus métricas y su umbral epidémico, y simula un proceso SIR sobre cada una. Las funciones auxiliares están en `redes.py`.

## Cómo Ejecutar

Requisitos: Python 3.14 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Task 1.3:

```bash
python task1_3.py
```

Task 3:

```bash
jupyter notebook task3.ipynb
```
