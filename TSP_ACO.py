#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:01:12 2024

@author: pavan
"""

import random as r 


N = 6  
T = 100  
rho = 0.5  
m = 10  
Q = 1 
epsilon = 0.1  
alpha = 1  
beta = 1

r.seed(42)

distance_matrix = [[r.randint(1, 100) if i!=j else 0 for j in range(N)] for i in range(N)]
eta = [[1/distance_matrix[i][j]if i!=j else 0 for j in range(N)] for i in range(N)]

#pheromone levels
tau = [[epsilon for i in range(N)]for i in range(N)]

#Step1
#stocastically constructing a solution
def select_next_destination(cur_city, tau, eta, alpha, beta, unvisited):
    prob = []
    summation_denominator = sum([(tau[cur_city][j] ** alpha) * (eta[cur_city][j] ** beta) for j in unvisited])
    
    for j in unvisited:
        p = ((tau[cur_city][j] ** alpha) * (eta[cur_city][j] ** beta)) / summation_denominator
        prob.append(p)
    
    next_city = r.choices(unvisited,weights=prob, k=1)[0]
    return next_city

#Step 2
#updating the pheromones
def updating_pheromones(tau, delta_tau, rho, m):
    for x in range(len(tau)):
        for y in range(len(tau[x])):
            tau[x][y] = (1 - rho) * tau[x][y] + sum(delta_tau[z][x][y] for z in range(m))
            
def ACO_TSP(T, rho, m , Q, alpha, beta, epsilon, distance_matrix, N):
    best_path = None
    shortest_distance = float('inf')
    
    tau = [[epsilon for i in range(N)] for i in range(N)]
    
    for t in range(T):
        paths = []
        path_lengths = []
        delta_tau = [[[0 for i in range(N)]for i in range(N)]for i in range(m)]
        
        for k in range(m):
            unvisited = list(range(N))
            #Starting from city 0
            cur_city = 0
            #starting from city random
            #cur_city = r.choice(unvisited)
            unvisited.remove(cur_city)
            trip = [cur_city]
            
            while unvisited:
                next_city = select_next_destination(cur_city, tau, eta, alpha, beta, unvisited)
                trip.append(next_city)
                unvisited.remove(next_city)
                cur_city = next_city
                
            #returning to the starting point
            trip_length = sum([distance_matrix[trip[i]][trip[i+1]] for i in range(N - 1)])
            trip_length += distance_matrix[trip[-1]][trip[0]]
            
            paths.append(trip)
            path_lengths.append(trip_length)
            
            #updating pheromone for this trip
            for i in range(N-1):
                delta_tau[k][trip[i]][trip[i+1]] = Q / trip_length
            delta_tau[k][trip[-1]][trip[0]] = Q / trip_length
            
        min_distance = min(path_lengths)
        min_path = paths[path_lengths.index(min_distance)]
        if min_distance < shortest_distance:
            shortest_distance = min_distance
            best_path = min_path
        
        updating_pheromones(tau, delta_tau, rho, m)
        print(f"Iteration {t + 1}: Best Length = {shortest_distance}, Best Solution = {best_path}")
        
    return best_path, shortest_distance

def aco_tsp_backtracking(distance_matrix):
    n = len(distance_matrix)
    visited = [False] * n
    best_trip = []
    best_length = float('inf')
    
    def backtrack(cur_city, c, cur_distance, trip):
        nonlocal best_length, best_trip
        
        #all cities visited
        if c == n and distance_matrix[cur_city][trip[0]] >0:
            total_distance = cur_distance + distance_matrix[cur_city][trip[0]]
            if total_distance < best_length:
                best_length = total_distance
                best_trip = trip[:]
            return
        
        for nc in range(n):
            if not visited[nc] and distance_matrix[cur_city][nc] > 0:
                visited[nc] = True
                trip.append(nc)
                
                if cur_distance + distance_matrix[cur_city][nc] < best_length:
                    backtrack(nc, c+1, cur_distance+distance_matrix[cur_city][nc], trip)
                
                visited[nc] = False
                trip.pop()
    
    visited[0] = True
    backtrack(0, 1, 0, [0])
    
    
    return best_trip, best_length

best_solution, best_length = ACO_TSP(T, rho, m, Q, alpha, beta, epsilon, distance_matrix, N)

bt_solution, bt_length = aco_tsp_backtracking(distance_matrix)

print("\nBest ACO Solution:", best_solution)
print("Best ACO Length:", best_length)
print("\nBacktracking Solution:", bt_solution)
print("Backtracking Length:", bt_length)
                
