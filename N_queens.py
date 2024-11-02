#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:09:59 2024

@author: pavan
"""

import random as r 

def fitness(queen_positions):
    n = len(queen_positions)
    clash = 0
    for i in range(n):
        for j in range(i+1, n):
            if queen_positions[i] == queen_positions[j] or abs(queen_positions[i] - queen_positions[j]) == abs(i-j):
                clash +=1
    return -clash

def crossover(parent1, parent2):
    n = len(parent1)
    child1, child2 = [-1]*n, [-1]*n

    start, end = sorted(r.sample(range(n), 2))

    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    def fill_child(child, parent):
        available = [x for x in parent if x not in child]
        for i in range(n):
            if child[i] == -1:
                child[i] = available.pop(0)

    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return child1, child2

def mutation(queen_positions):
    n = len(queen_positions)
    x, y = r.sample(range(n), 2)
    queen_positions[x], queen_positions[y] = queen_positions[y], queen_positions[x]
    
def selection(population, fitness_score):
    size = 3
    fitness_selection = r.choices(population, k = size)
    fitness_selected = [fitness(i) for i in fitness_selection]
    selected_value = fitness_selection[fitness_selected.index(max(fitness_selected))]
    
    return selected_value

def EA(n, population_size, generations):
    new_population = []
    population = [r.sample(range(n), n) for i in range(population_size)]
    for gen in range(generations):
        fitness_scores = [fitness(x) for x in population]
        best_fitness = max(fitness_scores)
        print(f"Generation {gen}: Best fitness = {-best_fitness}")
        
        if best_fitness == 0:
            break
        
        while len(new_population) < population_size:
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            
            if r.random() < 0.8:
                child1 , child2 = crossover(parent1, parent2)
            else:
                child1,child2 = parent1[:], parent2[:]
            
            if r.random() < 0.2:
                mutation(child1)
            
            if r.random() < 0.2:
                mutation(child2)
                
            new_population.append(child1)
            new_population.append(child2)
        
        population = new_population
    
    N_queen = population[fitness_scores.index(best_fitness)]
    return N_queen

N= 8
N_queen_puzzle =EA(N, 100, 1000)
print("N Queens puzzle solution is found\nOptimal Solution: ", N_queen_puzzle)
               