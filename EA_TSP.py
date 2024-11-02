#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:27:05 2024

@author: pavan
"""

import numpy as np
import matplotlib.pyplot as plt
import random as r

def euclidean_distance(x,y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def fitness(tsp, cities):
    total_distance = 0
    for i in range(len(tsp)):
        total_distance += euclidean_distance(cities[tsp[i-1]], cities[tsp[i]])
    return 1/total_distance

def crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(r.sample(range(size), 2))
    child[start:end + 1] = parent1[start:end + 1]

    pos = end + 1
    for city in parent2:
        if city not in child:
            if pos == size:
                pos = 0
            child[pos] = city
            pos += 1

    return child

def mutation(tsp):
    size = len(tsp)
    x, y = r.sample(range(size), 2)
    tsp[x], tsp[y] = tsp[y], tsp[x]
    return tsp

def selection(population, cities, size = 5):
    selected = r.sample(population, size)
    selected = sorted(selected, key = lambda x: fitness(x, cities), reverse=True)
    return selected[0]

def EA_TSP(cities, population_size, generations, crossover_prob, mutation_prob):
    pop = []
    no_cities = len(cities)
    
    for i in range(population_size):
        #Starting from city 0
        individual = [0]  
        individual += r.sample(range(1, no_cities), no_cities - 1)
        #Starting from city random
        # val = list(np.random.permutation(no_cities))
        # pop.append(val)
        pop.append(individual)
        
    best_fitness = -1
    best_soln = None
    fitness_scores = []
    
    for gen in range(generations):
        new_pop = []
        for i in range(population_size):
            parent1 = selection(pop, cities)
            parent2 = selection(pop, cities)
            
            if r.random() < crossover_prob:
                child = crossover(parent1, parent2)
            else:
                child = parent1[:]
            
            if r.random() < mutation_prob:
                child = mutation(child)
            
            new_pop.append(child)
        
        pop = new_pop
        cur_best = max(pop, key= lambda x: fitness(x, cities)) 
        cur_fitness = fitness(cur_best, cities)
        
        if cur_fitness > best_fitness:
            best_fitness = cur_fitness
            best_soln = cur_best
        
        fitness_scores.append(best_fitness)
        print(f'Generation {gen+1}: Best fitness = {best_fitness}')
    
    return best_soln, 1/best_fitness, fitness_scores

def tsp_backtrack(cities, visited, current_path, current_distance, best_solution, best_distance):
    num_cities = len(cities)
    
    if len(current_path) == num_cities:
        current_distance += euclidean_distance(cities[current_path[0]], cities[current_path[-1]])
        if current_distance < best_distance[0]:
            best_distance[0] = current_distance
            best_solution[:] = current_path[:]
        return

    for i in range(num_cities):
        if not visited[i]:
            visited[i] = True
            current_path.append(i)
            next_distance = current_distance + euclidean_distance(cities[current_path[-2]], cities[i]) if len(current_path) > 1 else 0
            if next_distance < best_distance[0]:
                tsp_backtrack(cities, visited, current_path, next_distance, best_solution, best_distance)
            
            # Undo the move
            visited[i] = False
            current_path.pop()
            
def tsp_brute_force(cities):
    num_cities = len(cities)
    
    visited = [False] * num_cities
    best_solution = []
    best_distance = [float('inf')]
    for i in range(num_cities):
        visited[i] = True
        tsp_backtrack(cities, visited, [i], 0, best_solution, best_distance)
        visited[i] = False
    
    return best_solution, best_distance[0]

def visualize_tsp(cities, solution):
    solution = list(solution) + [solution[0]]
    path = np.array([cities[i] for i in solution])

    plt.plot(path[:, 0], path[:, 1], 'o-', label='Path')
    plt.scatter(path[:, 0], path[:, 1], color='red', s=50, zorder=5)
    plt.title('TSP Solution')
    plt.show()
 
def read_tsp_file(file_path):
    cities = []
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    for line in data[8:]:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 3:
                _, x, y = parts
                cities.append((float(x), float(y)))
    return np.array(cities)

N=4
#cities = np.array([(1,0,0), (2,1,1), (3,0,1), (4,1,0)])
cities = np.array([(0, 0), (1, 1), (0, 1), (1, 0)])
#cities = read_tsp_file("/Users/pavan/Desktop/AI HW5/xqf131.tsp")
#cities = read_tsp_file("/Users/pavan/Desktop/AI HW5/xqg237.tsp")
best_solution, best_distance, fitness_scores = EA_TSP(cities, 100, 1000, 0.9, 0.1)

print(f"Best solution: {best_solution}")
print(f"Best distance: {best_distance}")

visualize_tsp(cities, best_solution)
            
optimal_solution, optimal_distance = tsp_brute_force(cities)
print(f"Optimal solution (brute force): {optimal_solution}")
print(f"Optimal distance: {optimal_distance}")                