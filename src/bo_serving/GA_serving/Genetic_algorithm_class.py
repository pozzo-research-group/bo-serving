import math
import numpy as np
import random

class ExperimentData():
    """"
    Fancy cache for experimental data 
    """

    def __init__(self, n_params):
        self.data = None
        self.n_params = n_params
        self.open_trials = []
        self.n_complete_trials = 0
        self.extra_data = {}
        

        # X_data is a dict of {trial_index:{sample_id:}}
        # store data in a dict of {trial_index:{sample_id:{x_params:[list], y_value:float}}}

    def complete_trial(self, trial_index, new_y_data, extra_data = None):
        """
        Add results from a new trial 

        new_x_params: torch tensor or np array nx1
        new_y_data: str, float, or int
        """

        assert trial_index in self.open_trials, 'Trial index is not in set of open trials'
        assert isinstance(new_y_data, list)

        try:
            new_y_data = [float(yval) for yval in new_y_data]
        except:
            raise ValueError(f'Could not cast new_y_data {new_y_data} as float')
        

       
        trial_data = self.data[trial_index]

        for i, sample in enumerate(trial_data.values()):
            sample['y_value'] = new_y_data[i]

        trial_data['extra_data'] = extra_data

        self.data[trial_index] = trial_data


        # close out trial bookkeeping
        self.n_complete_trials += 1
        self.open_trials.remove(trial_index)


    def get_all_data(self):
        return self.data
    
    def get_trial_data(self, trial_index):
        return self.data[trial_index]
    
    def get_population(self, trial_index):
        trial_data = self.data[trial_index]
        params = []
        for i in len(trial_data.items()) - 1:
            params.append(['x_values'])

        return params
    
    def get_population_fitness(self, trial_index):
        trial_data = self.data[trial_index]
        fitness = []
        for key, value in trial_data.items():
            fitness.append(value['y_value'])

        return fitness
   
    
    def register_new_trial(self, parameterization):
    
        assert len(parameterization[0]) == self.n_params, f'Parameterization must have {self.n_params} entries'

        if self.data is None:
            trial_index = 0
            self.data = {}
        elif len(self.open_trials) == 0:
            trial_index = len(self.data)
        else:
            trial_index = max(list(self.open_trials.keys())) + 1

        trial_data = {}
        for i, params in enumerate(parameterization):
            trial_data[i] = {'x_values':params, 'y_value':None}
        

        self.data[trial_index] = trial_data

        self.open_trials.append(trial_index)

        return trial_index

class GeneticAlgorithm():

    def __init__(self, target_color, population_size, n_params, initial_max_distance = 50, color_thresh = 1.73, r_cross = 0.5, r_mut = 0.01):
        
        self.population_size = population_size
        self.n_params = n_params
        self.initial_max_distance = initial_max_distance
        self.ExperimentData = ExperimentData(n_params)
        self.color_tresh = color_thresh
        self.target_color = target_color
        self.r_cross = r_cross
        self.r_mut = r_mut
        self.initialized = False
        return
    
    def update(self, trial_index, distances, extra_data = None):
        """
        Takes results from a completed experiment and registers them with the data store. Don't produce next generation yet. 
        """

        self.ExperimentData.complete_trial(trial_index, distances, extra_data=extra_data)
        
        return
    
    def get_next_generation(self):
        
        # Initial population of random weights with a distance constraint
        if self.initialized == False:
            new_pop = self.random_population()
            self.initialized = True
        
        else:
            
            most_recent_trial = self.ExperimentData.n_complete_trials - 1
            pop = self.ExperimentData.get_population(most_recent_trial)
            fitness = self.ExperimentData.get_population_fitness(most_recent_trial)

            best, best_eval = None, float('inf')
            
            # Check for new best solution
            for i in range(len(pop)):
                if fitness[i] < best_eval:
                    best, best_eval = pop[i], fitness[i]
                    #print(f">Gen {most_recent_trial}, new best: {mix_colors(pop[i])} - Weights: {pop[i]} - Distance: {scores[i]:.2f}")
                    if best_eval <= self.color_thresh:
                        print(f"Close enough solution found with distance <= {self.color_thresh}!")
                        return best, best_eval
                    if best_eval == 0:
                        print("Perfect solution found!")
                        return best, best_eval
            
            # Update the max_distance to the best_eval of the current generation
            self.max_distance = best_eval
            
            # Perform tournament selection to create a new generation of parents
            parents = self.tournament_selection(pop, self.target_color, num_tournaments=self.population_size)
            
            # Create the next generation through crossover and mutation
            children = []
            for i in range(0, self.population_size, 2):
                # Get selected parents in pairs
                p1, p2 = parents[i], parents[i+1]
                # Crossover and mutation
                for child in self.crossover(p1, p2):
                    self.mutation(child)
                    # Ensure no duplicates in the new generation
                    while child in children: # or distance(mix_colors(child), target_color) > max_distance:
                        child = self.random_population()
                       #while distance(mix_colors(child), target_color) > max_distance:
                       #     child = generate_random_weights()
                    children.append(child)
            
            # Replace population
            new_pop = children
        #self.print_population(new_pop, self.target_color)

        trial_index = self.ExperimentData.register_new_trial(new_pop)
            
        return new_pop, trial_index
    

    #def mix_colors(self, weights):
    #    z_mix = [0] * expected_length
    #    for i in range(expected_length):
    #        z_mix[i] = (weights[0] * z1[i] +
    #                    weights[1] * z2[i] +
    #                    weights[2] * z3[i] +
    #                    weights[3] * z4[i] +
    #                    weights[4] * z5[i])
    #    rgb_mix = mixbox.latent_to_rgb(z_mix)
    #    return rgb_mix

# Calculate the Euclidean distance between the target color and the generated color
    def distance(self, color, target_color):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(color, target_color)))

    # Create a population of random weights with a distance constraint
    def random_population(self):
        population = []

        while len(population) < self.population_size:
            weights = self.generate_random_weights()
            population.append(weights)

        return population
    
    def generate_random_weights(self):
        weights = [random.random() for _ in range(self.n_params)]
        total = sum(weights)
        return [w / total for w in weights]  # Normalize to sum to 1

    # Use the distance as the fitness score, the lower the better
    #def fitness(self, weights, target_color):
    #    color = self.mix_colors(weights)
    #    return self.distance(color, target_color)

    # Print the RGB values and fitness scores of the colors in the population
    def print_population(self, population, target_color):
        for idx, weights in enumerate(population):
            color = self.mix_colors(weights)
            score = self.fitness(weights, target_color)
            print(f"Color {idx + 1}: {color} - Weights: {weights} - Distance: {score:.2f}")

    # Perform tournament selection to create a new generation of parents
    def tournament_selection(self, population, population_fitness, target_color, num_tournaments):
        parents = []
        for _ in range(num_tournaments):
            # Select two random individuals
            index_1 = random.randint(0, len(population))
            index_2 = random.randint(0, len(population))


            weights1 = population[index_1]
            weights2 = population[index_2]
            
            # Compare their fitness scores
            score1 = population_fitness[index_1]
            score2 = population_fitness[index_2]
            
            # Select the individual with the lower score (minimizing distance)
            if score1 <= score2:
                parents.append(weights1)
            else:
                parents.append(weights2)
        
        return parents

    # Perform uniform crossover to create a new generation of children
    def crossover(self, parent1, parent2):
        child1, child2 = [], []
        for i in range(len(parent1)):  # For each weight
            if random.random() < self.r_cross:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return [child1, child2]

    # Perform mutation on the children
    def mutation(self, child):
        for i in range(len(child)):  # For each weight
            if random.random() < self.r_mut:  # mutate the weight
                child[i] = random.random()
        total = sum(child)
        for i in range(len(child)):
            child[i] /= total  # Normalize to sum to 1

        return