import random
import numpy as np

# ----------------------
#  Datos del problema
# ----------------------
num_parties = 5
num_entities = 50

# Reparto aleatorio original de curules entre partidos (conteo fijo)
# Generamos lista de 50 asientos (curules) y contamos cuántos tiene cada partido
seats = [random.randrange(num_parties) for _ in range(num_entities)]
seat_counts = [seats.count(p) for p in range(num_parties)]

# Lista de entidades y pesos aleatorios [1,100]
entities = [f"Ent{i+1}" for i in range(num_entities)]
weights = [random.randint(1,100) for _ in range(num_entities)]

# Participación de poder objetivo proporcional a curules
# Si un partido tiene más curules, debe recibir más poder
total_weight = sum(weights)
target_share = [count / num_entities * total_weight for count in seat_counts]

# ---------------------------------
#  Función de reparación para mantener conteos
# ---------------------------------
def repair(individual):
    # Asegura que cada partido reciba exactamente tantas entidades como curules tiene
    counts = [individual.count(p) for p in range(num_parties)]
    excess = {p: counts[p] - seat_counts[p] for p in range(num_parties) if counts[p] > seat_counts[p]}
    deficit = {p: seat_counts[p] - counts[p] for p in range(num_parties) if counts[p] < seat_counts[p]}
    for p in list(excess):
        while excess[p] > 0:
            idx = individual.index(p)
            q = random.choice([d for d, v in deficit.items() if v > 0])
            individual[idx] = q
            excess[p] -= 1
            deficit[q] -= 1
    return individual

# Función de aptitud: mide la diferencia entre lo asignado y lo deseado
# Se busca minimizar esta diferencia (error cuadrático)
def fitness(ind):
    assigned = [0] * num_parties
    for i, party in enumerate(ind):
        assigned[party] += weights[i]
    return sum((assigned[p] - target_share[p]) ** 2 for p in range(num_parties))

# ---------------------------------
#  Algoritmo Genético
# ---------------------------------
generations = 200
pop_size = 100
tourn_k = 3
cx_prob = 0.8
mut_prob = 0.2

# Genera individuo inicial: asigna entidades a partidos según número de curules
def rand_individual():
    base = []
    for p, c in enumerate(seat_counts):
        base += [p] * c
    random.shuffle(base)
    return base

# Inicialización de la población
pop = [rand_individual() for _ in range(pop_size)]

for gen in range(generations):
    # Evaluación
    scores = [fitness(ind) for ind in pop]

    # Selección por torneo
    selected = []
    for _ in range(pop_size):
        aspirants = random.sample(range(pop_size), tourn_k)
        winner = min(aspirants, key=lambda i: scores[i])
        selected.append(pop[winner][:])

    # Cruce de un punto con reparación para mantener número de entidades por partido
    offspring = []
    for i in range(0, pop_size, 2):
        p1, p2 = selected[i], selected[i + 1]
        if random.random() < cx_prob:
            pt = random.randint(1, num_entities - 1)
            c1 = repair(p1[:pt] + p2[pt:])
            c2 = repair(p2[:pt] + p1[pt:])
        else:
            c1, c2 = p1[:], p2[:]
        offspring += [c1, c2]

    # Mutación por intercambio entre dos entidades
    pop = []
    for ind in offspring:
        if random.random() < mut_prob:
            i, j = random.sample(range(num_entities), 2)
            ind[i], ind[j] = ind[j], ind[i]
        pop.append(repair(ind))

# Mejor solución final (menor error)
best = min(pop, key=fitness)
assigned_power = [0] * num_parties
for i, party in enumerate(best):
    assigned_power[party] += weights[i]

# Resultados
print("Asignación final de poder por partido (en puntos políticos):")
for p in range(num_parties):
    print(f"Partido {p+1}: {assigned_power[p]:.0f} puntos\t(meta: {target_share[p]:.1f})")

print("\nResumen:")

print(f"Error total cuadrático (fitness): {fitness(best):.2f}")
