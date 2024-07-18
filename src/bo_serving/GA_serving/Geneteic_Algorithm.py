'''
Best GA
Fixed:
using weights instead of rgb bit values
ensures latent_colors mix don't return index error
'''

import random
import math
import mixbox

# Function to generate random weights
def generate_random_weights():
    weights = [random.random() for _ in range(5)]
    total = sum(weights)
    return [w / total for w in weights]  # Normalize to sum to 1

# Define stock colors
rgb1 = (255, 0, 0)        # Red
rgb2 = (255, 255, 0)      # Yellow
rgb3 = (0, 0, 255)        # Blue
rgb4 = (255, 255, 255)    # White
rgb5 = (0, 0, 0)          # Black

# Convert RGB colors to latent space
z1 = mixbox.rgb_to_latent(rgb1)
z2 = mixbox.rgb_to_latent(rgb2)
z3 = mixbox.rgb_to_latent(rgb3)
z4 = mixbox.rgb_to_latent(rgb4)
z5 = mixbox.rgb_to_latent(rgb5)

latent_colors = [z1, z2, z3, z4, z5]

# Ensure latent vectors are the expected length
expected_length = 7
for i, z in enumerate(latent_colors, start=1):
    if len(z) != expected_length:
        raise ValueError(f"Latent vector z{i} has incorrect length {len(z)}, expected {expected_length}")

# Function to mix colors 
def mix_colors(weights):
    z_mix = [0] * expected_length
    for i in range(expected_length):
        z_mix[i] = (weights[0] * z1[i] +
                    weights[1] * z2[i] +
                    weights[2] * z3[i] +
                    weights[3] * z4[i] +
                    weights[4] * z5[i])
    rgb_mix = mixbox.latent_to_rgb(z_mix)
    return rgb_mix

# Calculate the Euclidean distance between the target color and the generated color
def distance(color, target_color):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(color, target_color)))

# Create a population of random weights with a distance constraint
def create_population(size, target_color, max_distance):
    population = []
    while len(population) < size:
        weights = generate_random_weights()
        #color = mix_colors(weights)
        #if distance(color, target_color) <= max_distance and color not in population:
        population.append(weights)
    return population

# Use the distance as the fitness score, the lower the better
def fitness(weights, target_color):
    color = mix_colors(weights)
    return distance(color, target_color)

# Print the RGB values and fitness scores of the colors in the population
def print_population(population, target_color):
    for idx, weights in enumerate(population):
        color = mix_colors(weights)
        score = fitness(weights, target_color)
        print(f"Color {idx + 1}: {color} - Weights: {weights} - Distance: {score:.2f}")

# Perform tournament selection to create a new generation of parents
def tournament_selection(population, target_color, num_tournaments):
    parents = []
    for _ in range(num_tournaments):
        # Select two random individuals
        weights1 = random.choice(population)
        weights2 = random.choice(population)
        
        # Compare their fitness scores
        score1 = fitness(weights1, target_color)
        score2 = fitness(weights2, target_color)
        
        # Select the individual with the lower score (minimizing distance)
        if score1 <= score2:
            parents.append(weights1)
        else:
            parents.append(weights2)
    
    return parents

# Perform uniform crossover to create a new generation of children
def crossover(parent1, parent2, r_cross):
    child1, child2 = [], []
    for i in range(len(parent1)):  # For each weight
        if random.random() < r_cross:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return [child1, child2]

# Perform mutation on the children
def mutation(child, r_mut):
    for i in range(len(child)):  # For each weight
        if random.random() < r_mut:  # mutate the weight
            child[i] = random.random()
    total = sum(child)
    for i in range(len(child)):
        child[i] /= total  # Normalize to sum to 1

# GA
def genetic_algorithm(target_color, max_iter, n_pop, r_cross, r_mut, initial_max_distance, color_thresh):
    print("Target Color:", target_color)
    
    # Initial population of random weights with a distance constraint
    pop = create_population(n_pop, target_color, initial_max_distance)
    
    best, best_eval = None, float('inf')
    max_distance = initial_max_distance
    
    for gen in range(max_iter):
        print(f"\nGeneration {gen+1}")
        print_population(pop, target_color)
        
        # Evaluate all candidates in the population
        scores = [fitness(c, target_color) for c in pop]
        
        # Check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(f">Gen {gen}, new best: {mix_colors(pop[i])} - Weights: {pop[i]} - Distance: {scores[i]:.2f}")
                if best_eval <= color_thresh:
                    print(f"Close enough solution found with distance <= {color_thresh}!")
                    return best, best_eval
                if best_eval == 0:
                    print("Perfect solution found!")
                    return best, best_eval
        
        # Update the max_distance to the best_eval of the current generation
        max_distance = best_eval
        
        # Perform tournament selection to create a new generation of parents
        parents = tournament_selection(pop, target_color, num_tournaments=n_pop)
        
        # Create the next generation through crossover and mutation
        children = []
        for i in range(0, n_pop, 2):
            # Get selected parents in pairs
            p1, p2 = parents[i], parents[i+1]
            # Crossover and mutation
            for child in crossover(p1, p2, r_cross):
                mutation(child, r_mut)
                # Ensure no duplicates in the new generation
                while child in children or distance(mix_colors(child), target_color) > max_distance:
                    child = generate_random_weights()
                    while distance(mix_colors(child), target_color) > max_distance:
                        child = generate_random_weights()
                children.append(child)
        
        # Replace population
        pop = children
        
    return best, best_eval

# Define the target color
target_color = [179, 36, 40]

# Parameters for the GA
max_iter = 24  # Maximum number of iterations
n_pop = 4      # Population size
r_cross = 0.5   # Crossover rate
r_mut = 0.08    # Mutation rate
initial_max_distance = 50  # Initial maximum distance for initial population
color_thresh = 2   # Threshold for stopping the algorithm if a close enough solution is found: sqrt(3)

# Run the genetic algorithm
best_weights, best_score = genetic_algorithm(target_color, max_iter, n_pop, r_cross, r_mut, initial_max_distance, color_thresh)

# Print the final best solution
print("\nBest Solution Found:")
best_color = mix_colors(best_weights)
print(f"Color: {best_color} - Weights: {best_weights} - Distance: {best_score:.2f}")

'''
Checking Validity in Weight Values:


'''
