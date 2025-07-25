import numpy as np
from scipy.optimize import linprog

# ----------------------
#  Datos del problema
# ----------------------
# Lista de plantas y ciudades
plantas  = ["C", "B1", "M", "B2"]
ciudades = ["Cali", "Bogotá", "Medellín", "Barranquilla"]

# Capacidades máximas de cada planta (GW/día)
capacidad = np.array([3, 6, 5, 4])
# Demandas mínimas de cada ciudad (GW/día)
demanda   = np.array([4, 3, 5, 3])

# Matriz de costos de transporte $/GW: c_trans[i,j]
c_trans = np.array([
    [1, 4, 3, 6],  # desde planta C
    [4, 1, 4, 5],  # desde planta B1
    [3, 4, 1, 4],  # desde planta M
    [6, 5, 4, 1],  # desde planta B2
])
# Vector de costos de generación $/GW: c_gen[i]
c_gen = np.array([680, 720, 660, 750])

# Número total de variables x_ij (16 flujos)
N = 4 * 4

# ---------------------------------
#  1) Construcción de la función objetivo
# ---------------------------------
# Aplanamos c_trans y sumamos c_gen para cada flujo
# Resultado: vector C de dimensión N
C = c_trans.flatten() + np.repeat(c_gen, 4)

# ---------------------------------
#  2) Restricciones de capacidad
# ---------------------------------
# Cada planta i no puede despachar más de capacidad[i]
A_ub = np.zeros((4, N))  # izquierda de <=
b_ub = capacidad.copy()   # derecho de <=
for i in range(4):
    for j in range(4):
        # x[i*4+j] corresponde a planta i → ciudad j
        A_ub[i, i * 4 + j] = 1

# ---------------------------------
#  3) Restricciones de demanda
# ---------------------------------
# Cada ciudad j debe recibir al menos demanda[j]
# Convertimos a forma estándar: -sum_i x_ij <= -demanda[j]
A_ub2 = np.zeros((4, N))  # izquierda de <= para demanda
b_ub2 = -demanda.copy()  # derecho de <= (negativo)
for j in range(4):
    for i in range(4):
        A_ub2[j, i * 4 + j] = -1

# Combinamos restricciones de oferta y demanda
aA_ub = np.vstack([A_ub, A_ub2])
bb_ub = np.hstack([b_ub, b_ub2])

# ---------------------------------
#  4) Límites de las variables
# ---------------------------------
# Todas las x_ij >= 0
bounds = [(0, None)] * N

# ---------------------------------
#  5) Resolución con linprog
# ---------------------------------
res = linprog(
    C,
    A_ub=aA_ub,
    b_ub=bb_ub,
    bounds=bounds,
    method="highs"
)

# ---------------------------------
#  6) Impresión de resultados
# ---------------------------------
if res.success:
    x = res.x.reshape((4, 4))  # reconstruimos matriz 4x4
    print("Despacho óptimo (GW):")
    for i in range(4):
        for j in range(4):
            if x[i, j] > 1e-6:  # solo flujos significativos
                print(f"Planta {plantas[i]} → Ciudad {ciudades[j]}: {x[i, j]:.0f} GW")
    print(f"\nCosto total: ${res.fun:,.2f}")
else:
    print("No se encontró solución:", res.message)
