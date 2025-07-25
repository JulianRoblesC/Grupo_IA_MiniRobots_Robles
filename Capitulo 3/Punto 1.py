import random
import math

# Parámetros del GA
generations = 100      # Número de generaciones
pop_size = 50          # Tamaño de población
tournament_k = 3       # Tamaño del torneo para selección
torch_prob = 0.8       # Probabilidad de cruce
mut_prob = 0.1         # Probabilidad de mutación
sigma = 0.1            # Desviación estándar para mutación gaussiana

# Función a maximizar: f(x) = x * sin(10πx) + 1, x ∈ [0,1]
def fitness(x):
    return x * math.sin(10 * math.pi * x) + 1

# 1) Inicialización: población de valores reales en [0,1]
population = [random.random() for _ in range(pop_size)]

for gen in range(generations):
    # 2) Evaluación: calculamos aptitud de cada individuo
    scores = [fitness(x) for x in population]

    # 3) Selección por torneo
    selected = []
    for _ in range(pop_size):
        aspirants = random.sample(range(pop_size), tournament_k)
        # Elegimos el mejor del subgrupo
        best = max(aspirants, key=lambda i: scores[i])
        selected.append(population[best])

    # 4) Cruce de padres por pares (arithmetic crossover)
    offspring = []
    for i in range(0, pop_size, 2):
        p1, p2 = selected[i], selected[i+1]
        if random.random() < torch_prob:
            alpha = random.random()
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = alpha * p2 + (1 - alpha) * p1
        else:
            c1, c2 = p1, p2
        offspring.extend([c1, c2])

    # 5) Mutación gaussiana y recorte a [0,1]
    population = []
    for x in offspring:
        if random.random() < mut_prob:
            x += random.gauss(0, sigma)
        # Aseguramos que x permanezca en [0,1]
        x = min(max(x, 0.0), 1.0)
        population.append(x)

# 6) Resultado: mejor individuo y su valor
best = max(population, key=fitness)
print(f"Mejor x: {best:.4f}")
print(f"Máximo f(x): {fitness(best):.4f}")
