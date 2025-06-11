# -*- coding: utf-8 -*-
"""
COIT29224 - Genetic Programming for Symbolic Regression
Definitive Version (v3.1 - Bug Fix)

This script implements a Genetic Programming (GP) system to solve a symbolic
regression problem. It is designed with a professional Object-Oriented structure,
includes comprehensive logging, and outputs data for an interactive visualisation.

Key Features:
- Object-Oriented Design: The GP process is encapsulated within a `GeneticProgramming` class.
- Structured Logging: Creates a `gp_evolution.log` file to record run details.
- Interactive Visualisation Output: Saves the best tree and fitness history to `gp_results.json`
  for visualisation with the accompanying `gp_visualiser.html` file.
- Pedagogical Comments: Code is commented to explain the 'why' behind the implementation,
  serving as an educational example.

Author: Sebastian R. (2025)
Course: COIT29224 - EEvolutionary Computation
Student ID: [Student ID Hidden for security]
Date: 06 June 2025
"""

import json
import logging
import math
import operator
import random
import time
import copy

# --- Configuration: GP Parameters ---
# These parameters control the evolutionary process. Tuning them can significantly
# impact performance and the quality of the solution.
GP_PARAMS = {
    "population_size": 300,        # Number of candidate solutions in each generation.
    "generations": 200,            # Maximum number of generations to run.
    "min_initial_depth": 2,        # Minimum depth for trees in the initial population.
    "max_initial_depth": 4,        # Maximum depth for trees in the initial population.
    "max_tree_depth": 10,          # Hard limit on tree depth to control bloat.
    "tournament_size": 5,          # Number of individuals in a selection tournament.
    "elitism_count": 3,            # Number of best individuals to carry over to the next generation.
    "crossover_rate": 0.8,         # Probability of performing crossover.
    "mutation_rate": 0.25,         # Probability an offspring undergoes mutation.
    "desired_fitness_threshold": 0.05, # A target MSE to stop evolution early if reached.
    "constant_range": (-20.0, 20.0), # Range for generating float constants.
    "integer_constants": list(range(-20, 21)), # Pool of integer constants.
    "epsilon": 1e-6                # Small value for safe division.
}

# --- Setup: Logging Configuration ---
# Configures a logger to save run details to a file and print to console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gp_evolution.log", mode='w'), # Overwrites log each run
        logging.StreamHandler()
    ]
)

# --- Primitives: The Building Blocks of Expressions ---

def safe_div(x, y):
    """Protected division to prevent runtime errors from division by zero."""
    # Concept: Robustness. In GP, randomly generated expressions often attempt
    # invalid operations. Protecting against these is crucial for a stable run.
    # A small constant, epsilon, is used to check for near-zero denominators.
    if abs(y) < GP_PARAMS["epsilon"]:
        return 1.0  # Return a default, neutral value.
    return x / y

# The set of functions the GP can use as internal nodes in expression trees.
FUNCTIONS = {
    'add': operator.add, 'sub': operator.sub,
    'mul': operator.mul, 'div': safe_div
}
# The set of terminals (leaves) for the expression trees.
# 'const' is a placeholder for an ephemeral random constant.
TERMINALS = ['x', 'y', 'const']


class GPTree:
    """Represents a single expression tree, the core data structure in GP."""
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        # A unique ID for each node, useful for visualisation.
        self.id = random.randint(10000, 99999)

    def evaluate(self, x_val, y_val):
        """Recursively computes the output of the expression for given inputs."""
        # Concept: Recursive Evaluation. Tree structures are naturally processed
        # recursively. An operation at a node is performed on the results of
        # its children's own evaluations.
        if callable(self.data):
            # Guard against malformed trees during evolution.
            if self.left is None or self.right is None: return 1e9

            left_result = self.left.evaluate(x_val, y_val)
            right_result = self.right.evaluate(x_val, y_val)

            # Penalise expressions that result in non-standard numbers.
            if math.isinf(left_result) or math.isnan(left_result) or \
               math.isinf(right_result) or math.isnan(right_result):
                return 1e9

            try:
                result = self.data(left_result, right_result)
                if math.isinf(result) or math.isnan(result): return 1e9
                return result
            except (OverflowError, ValueError):
                return 1e9 # Return a high penalty for numerical issues.
        elif self.data == 'x':
            return float(x_val)
        elif self.data == 'y':
            return float(y_val)
        else:  # It's a constant terminal.
            return float(self.data)

    def build_random_tree(self, current_depth, max_depth, method="grow"):
        """Constructs a random tree using either the 'grow' or 'full' method."""
        # This method is central to the Ramped Half-and-Half initialisation.
        # 'full': Creates bushy trees by always choosing functions until max_depth.
        # 'grow': Creates varied shapes by allowing terminals before max_depth.
        if current_depth >= max_depth or \
           (method == "grow" and random.random() < 0.5 and current_depth >= GP_PARAMS["min_initial_depth"]):
            chosen_terminal = random.choice(TERMINALS)
            if chosen_terminal == 'const':
                const_type = random.choice(['float', 'int'])
                if const_type == 'float':
                    self.data = round(random.uniform(*GP_PARAMS["constant_range"]), 3)
                else:
                    self.data = float(random.choice(GP_PARAMS["integer_constants"]))
            else:
                self.data = chosen_terminal
            self.left, self.right = None, None
        else:
            self.data = random.choice(list(FUNCTIONS.values()))
            self.left = GPTree()
            self.left.build_random_tree(current_depth + 1, max_depth, method)
            self.right = GPTree()
            self.right.build_random_tree(current_depth + 1, max_depth, method)

    def get_node_count(self):
        """Recursively counts the total number of nodes in the tree."""
        if self.left is None and self.right is None: return 1
        return 1 + (self.left.get_node_count() if self.left else 0) + \
               (self.right.get_node_count() if self.right else 0)

    def get_depth(self, current_depth=0):
        """Recursively calculates the maximum depth of the tree."""
        if self.left is None and self.right is None: return current_depth
        left_depth = self.left.get_depth(current_depth + 1) if self.left else 0
        right_depth = self.right.get_depth(current_depth + 1) if self.right else 0
        return max(left_depth, right_depth)

    def collect_nodes_and_parents(self):
        """
        Collects all nodes and their parents using a breadth-first traversal.
        This is crucial for operators like crossover that need to modify the tree.
        Returns a list of (node, parent_node) tuples. The root's parent is None.
        """
        nodes = []
        queue = [(self, None)]  # Start with the root node and no parent.
        while queue:
            current_node, parent = queue.pop(0)
            nodes.append((current_node, parent))
            if current_node.left:
                queue.append((current_node.left, current_node))
            if current_node.right:
                queue.append((current_node.right, current_node))
        return nodes

    def to_dict(self):
        """Serialises the tree to a dictionary format for JSON output."""
        # This is essential for passing the tree structure to the HTML/JS visualiser.
        # It creates a hierarchical dictionary that mirrors the tree's shape.
        label = self._get_node_label()
        node_dict = {"name": label, "id": self.id, "children": []}
        if self.left:
            node_dict["children"].append(self.left.to_dict())
        if self.right:
            node_dict["children"].append(self.right.to_dict())
        return node_dict

    def _get_node_label(self):
        """Returns a string label for the node's data."""
        if callable(self.data):
            # Find the function's name (e.g., 'add') from its object reference.
            return [name for name, func in FUNCTIONS.items() if func == self.data][0]
        elif isinstance(self.data, float):
            return f"{self.data:.3f}"
        return str(self.data)

    def __str__(self):
        """Returns a LISP-style string representation of the tree."""
        # e.g., (add x (mul y 2.0))
        if self.left is None and self.right is None:
            return self._get_node_label()
        return f"({self._get_node_label()} {str(self.left)} {str(self.right)})"


class GeneticProgramming:
    """
    Encapsulates the entire GP evolutionary process.

    This class-based approach improves modularity and state management compared
    to a purely procedural script. It holds the population, dataset, and manages
    the generational loop.
    """
    def __init__(self, params, funcs, terms):
        self.params = params
        self.functions = funcs
        self.terminals = terms
        self.population = []
        self.dataset = self._get_target_dataset()
        self.best_overall_individual = None
        self.best_overall_fitness = float('inf')
        self.fitness_history = []

    def _get_target_dataset(self):
        """Constructs the dataset based on the assignment's Table 1."""
        dataset = []
        x_values, y_values = range(-1, 6), range(-1, 6)
        for x in x_values:
            for y in y_values:
                # The underlying formula deduced from the data table.
                if x == -1: res = (2 * y - 17) / 3.0
                else: res = (y + 6 * x - 12) / 3.0
                dataset.append(((float(x), float(y)), res))
        return dataset

    def _initialize_population(self):
        """Creates the initial population using structured ramped half-and-half."""
        # Rationale: A diverse initial population (in size and shape) is crucial
        # to ensure a broad exploration of the solution space from the outset.
        logging.info("Initializing population using ramped half-and-half...")
        pop = []
        depth_range = range(self.params["min_initial_depth"], self.params["max_initial_depth"] + 1)
        
        # Calculate how many individuals to generate for each depth and method.
        # This aims for an even distribution.
        inds_per_depth_method = self.params["population_size"] // (len(depth_range) * 2) if depth_range else 0

        for depth in depth_range:
            for _ in range(inds_per_depth_method):
                if len(pop) < self.params["population_size"]:
                    tree_full = GPTree()
                    tree_full.build_random_tree(0, depth, method="full")
                    pop.append(tree_full)
                
                if len(pop) < self.params["population_size"]:
                    tree_grow = GPTree()
                    tree_grow.build_random_tree(0, depth, method="grow")
                    pop.append(tree_grow)
        
        # Fill any remaining spots to meet population_size due to integer division.
        while len(pop) < self.params["population_size"]:
            depth = random.choice(depth_range)
            tree = GPTree()
            tree.build_random_tree(0, depth, method=random.choice(["full", "grow"]))
            pop.append(tree)

        self.population = pop
        logging.info(f"Initialized population with {len(self.population)} individuals.")


    def _calculate_fitness(self, individual):
        """Calculates the fitness of an individual using Mean Squared Error (MSE)."""
        # Fitness Concept: A quantitative measure of how well a solution performs.
        # MSE is used as it heavily penalises larger errors, driving the evolution
        # towards solutions that are consistently accurate.
        if not self.dataset: return float('inf')
        
        errors = []
        for (x,y), target in self.dataset:
            prediction = individual.evaluate(x,y)
            # Penalise non-numeric or extremely large values.
            if not isinstance(prediction, (int, float)) or math.isnan(prediction) or math.isinf(prediction):
                 errors.append(1e12)
            else:
                errors.append((prediction - target)**2)
        
        return sum(errors) / len(errors)

    def _tournament_selection(self, population_with_fitness):
        """Selects one parent using tournament selection."""
        # Selection Concept: Mimics 'survival of the fittest'. Fitter individuals
        # have a higher probability of being selected to pass on their genetic
        # material. Tournament selection is an efficient and balanced method.
        t_size = self.params["tournament_size"]
        # Ensure tournament is not larger than population.
        t_size = min(t_size, len(population_with_fitness)) 
        
        selected_contestants = random.sample(population_with_fitness, t_size)
        # The winner is the one with the lowest fitness (MSE).
        return min(selected_contestants, key=lambda item: item[0])[1]

    def _subtree_crossover(self, parent1, parent2):
        """Performs subtree crossover, creating two children."""
        # Crossover Concept: The primary recombination operator. It combines
        # building blocks (subtrees) from two parents, hoping to create
        # superior offspring. This version creates two children as specified.
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # The following calls now work because collect_nodes_and_parents is in GPTree
        nodes1 = [node for node, _ in child1.collect_nodes_and_parents()]
        nodes2 = [node for node, _ in child2.collect_nodes_and_parents()]

        if not nodes1 or not nodes2: return parent1, parent2

        # Select random subtrees from each parent to swap.
        graft1 = copy.deepcopy(random.choice(nodes2))
        graft2 = copy.deepcopy(random.choice(nodes1))

        # Perform the swap.
        self._replace_random_subtree(child1, graft1)
        self._replace_random_subtree(child2, graft2)
        
        # Bloat Control: Check depth and revert if necessary.
        if child1.get_depth() > self.params["max_tree_depth"]: child1 = parent1
        if child2.get_depth() > self.params["max_tree_depth"]: child2 = parent2

        return child1, child2

    def _replace_random_subtree(self, tree, graft):
        """Helper for crossover: replaces a random node in `tree` with `graft`."""
        nodes_with_parents = tree.collect_nodes_and_parents()
        if not nodes_with_parents: return
        
        # Choose a random node from the tree to be the point of replacement.
        node_to_replace, parent = random.choice(nodes_with_parents)
        
        if parent is None: # Replacing the root of the tree.
            tree.data, tree.left, tree.right, tree.id = graft.data, graft.left, graft.right, graft.id
        elif parent.left == node_to_replace:
            parent.left = graft
        else:
            parent.right = graft


    def _point_mutation(self, individual):
        """Performs point mutation on an individual."""
        # Mutation Concept: Introduces random variation, preventing premature
        # convergence and allowing for fine-tuning of solutions.
        mutant = copy.deepcopy(individual)
        nodes = [node for node, _ in mutant.collect_nodes_and_parents()]
        if not nodes: return individual

        node_to_mutate = random.choice(nodes)

        if callable(node_to_mutate.data): # It's a function node
            current_func = node_to_mutate.data
            # Exclude current function from choices to ensure a change.
            new_func = random.choice([f for f in self.functions.values() if f != current_func])
            node_to_mutate.data = new_func
        else: # It's a terminal node
            # Allow mutation to change a constant's value or change its type.
            if isinstance(node_to_mutate.data, (int, float)) and random.random() < 0.5:
                # Perturb the constant's value by a small amount.
                perturbation = random.uniform(-0.5, 0.5)
                node_to_mutate.data = round(node_to_mutate.data + perturbation, 3)
            else:
                # Change to a new random terminal.
                new_terminal = random.choice(self.terminals)
                if new_terminal == 'const':
                     node_to_mutate.data = round(random.uniform(*self.params["constant_range"]), 3)
                else:
                    node_to_mutate.data = new_terminal
        return mutant

    def run(self):
        """Executes the main GP evolutionary loop."""
        logging.info("Starting Genetic Programming evolution...")
        logging.info(f"Parameters: {json.dumps(self.params, indent=2)}")

        self._initialize_population()

        for gen in range(1, self.params["generations"] + 1):
            # 1. Evaluate Fitness
            population_with_fitness = [
                (self._calculate_fitness(ind), ind) for ind in self.population
            ]
            population_with_fitness.sort(key=lambda item: item[0])
            
            # 2. Update Best Solution
            current_best_fitness = population_with_fitness[0][0]
            self.fitness_history.append(current_best_fitness)
            
            if current_best_fitness < self.best_overall_fitness:
                self.best_overall_fitness = current_best_fitness
                self.best_overall_individual = copy.deepcopy(population_with_fitness[0][1])
                logging.info(
                    f"Generation {gen:03d}: New best! Fitness (MSE) = {self.best_overall_fitness:.6f}, "
                    f"Nodes = {self.best_overall_individual.get_node_count()}"
                )
            else:
                 avg_fitness = sum(f for f, _ in population_with_fitness) / len(population_with_fitness)
                 logging.info(
                    f"Generation {gen:03d}: Best this gen MSE = {current_best_fitness:.6f} "
                    f"(Avg: {avg_fitness:.2f}, Overall best: {self.best_overall_fitness:.6f})"
                 )

            # 3. Check Termination
            if self.best_overall_fitness <= self.params["desired_fitness_threshold"]:
                logging.info(f"Desired fitness threshold reached at generation {gen}.")
                break
            
            if gen == self.params["generations"]:
                logging.info("Maximum generations reached.")
                # No break here, allow final generation to complete for logging.
            
            if gen > self.params["generations"]:
                break

            # 4. Create New Generation
            new_population = []
            
            # Elitism: Carry over the best individuals.
            # Ensure not to select more elites than exist.
            num_elites = min(self.params["elitism_count"], len(population_with_fitness))
            for i in range(num_elites):
                new_population.append(copy.deepcopy(population_with_fitness[i][1]))
            
            # Reproduction Loop
            while len(new_population) < self.params["population_size"]:
                parent1 = self._tournament_selection(population_with_fitness)
                parent2 = self._tournament_selection(population_with_fitness)
                
                # Crossover
                if random.random() < self.params["crossover_rate"]:
                    offspring1, offspring2 = self._subtree_crossover(parent1, parent2)
                else: # Reproduction (cloning)
                    offspring1, offspring2 = parent1, parent2
                
                # Mutation
                if random.random() < self.params["mutation_rate"]:
                    offspring1 = self._point_mutation(offspring1)
                if random.random() < self.params["mutation_rate"]:
                    offspring2 = self._point_mutation(offspring2)

                new_population.append(offspring1)
                if len(new_population) < self.params["population_size"]:
                    new_population.append(offspring2)

            self.population = new_population

        logging.info("Evolutionary process ended.")
        self._log_and_save_results()

    def _log_and_save_results(self):
        """Logs the final results and saves them to a JSON file."""
        if self.best_overall_individual:
            logging.info("\n--- Final Results ---")
            logging.info(f"Best Fitness (MSE): {self.best_overall_fitness:.8f}")
            logging.info(f"Best Individual (Prefix): {str(self.best_overall_individual)}")
            logging.info(f"Depth: {self.best_overall_individual.get_depth()}, Nodes: {self.best_overall_individual.get_node_count()}")
            
            # Save results for visualisation
            results_data = {
                "best_tree": self.best_overall_individual.to_dict(),
                "fitness_history": self.fitness_history,
                "final_mse": self.best_overall_fitness,
                "params": self.params
            }
            try:
                with open("gp_results.json", "w") as f:
                    json.dump(results_data, f, indent=4)
                logging.info("Results saved to gp_results.json for visualisation.")
            except Exception as e:
                logging.error(f"Could not save results to JSON file: {e}")
            
        else:
            logging.info("No solution was found.")

if __name__ == '__main__':
    start_time = time.time()
    # Create an instance of the GP system and run it.
    gp_system = GeneticProgramming(GP_PARAMS, FUNCTIONS, TERMINALS)
    gp_system.run()
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")

