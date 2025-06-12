# -*- coding: utf-8 -*-
"""
COIT29224 - Genetic Programming for Symbolic Regression
Definitive Version (v4.0 - Final Bug Fixes)

This script implements a Genetic Programming (GP) system to solve a symbolic
regression problem. It is designed with a professional Object-Oriented structure,
includes comprehensive logging, and outputs data for an interactive visualisation.

Key Features:
- Object-Oriented Design: The GP process is encapsulated within a `GeneticProgramming` class.
- Timestamped Logging: Creates a unique, timestamped log file for each run.
- Interactive Visualisation Output: Saves the best tree and fitness history to `gp_results.json`
  for visualisation with the accompanying `gp_visualiser.html` file.
- Pedagogical Comments: Code is commented to explain the 'why' behind the implementation,
  serving as an educational example.

Author: Sebastian R. (2025)
Course: COIT29224 - EEvolutionary Computation
Student ID: [Student ID Hidden for security]
Date: 13 June 2025
"""

import json
import logging
from datetime import datetime
import math
import operator
import random
import time
import copy

# --- Configuration: GP Parameters ---
GP_PARAMS = {
    "population_size": 300,
    "generations": 200,
    "min_initial_depth": 2,
    "max_initial_depth": 4,
    "max_tree_depth": 10,
    "tournament_size": 5,
    "elitism_count": 3,
    "crossover_rate": 0.8,
    "mutation_rate": 0.25,
    "desired_fitness_threshold": 0.05,
    "constant_range": (-20.0, 20.0),
    "integer_constants": list(range(-20, 21)),
    "epsilon": 1e-6
}

# --- Setup: Logging Configuration ---
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"gp_run_{run_timestamp}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_filename, mode='w')
console_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# --- Primitives: The Building Blocks of Expressions ---
def safe_div(x, y):
    if abs(y) < GP_PARAMS["epsilon"]: return 1.0
    return x / y

FUNCTIONS = {
    'add': operator.add, 'sub': operator.sub,
    'mul': operator.mul, 'div': safe_div
}
TERMINALS = ['x', 'y', 'const']


class GPTree:
    """Represents a single expression tree, the core data structure in GP."""
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def evaluate(self, x_val, y_val):
        if callable(self.data):
            if self.left is None or self.right is None: return 1e9
            left_result = self.left.evaluate(x_val, y_val)
            right_result = self.right.evaluate(x_val, y_val)

            if math.isinf(left_result) or math.isnan(left_result) or \
               math.isinf(right_result) or math.isnan(right_result):
                return 1e9

            try:
                result = self.data(left_result, right_result)
                if math.isinf(result) or math.isnan(result): return 1e9
                return result
            except (OverflowError, ValueError):
                return 1e9
        elif self.data == 'x':
            return float(x_val)
        elif self.data == 'y':
            return float(y_val)
        else:
            return float(self.data)

    def build_random_tree(self, current_depth, max_depth, method="grow"):
        if current_depth >= max_depth or \
           (method == "grow" and random.random() < 0.5 and current_depth >= GP_PARAMS["min_initial_depth"]):
            chosen_terminal = random.choice(TERMINALS)
            if chosen_terminal == 'const':
                if random.random() < 0.7:
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
        if self.left is None and self.right is None: return 1
        return 1 + (self.left.get_node_count() if self.left else 0) + \
               (self.right.get_node_count() if self.right else 0)

    def get_depth(self, current_depth=0):
        if self.left is None and self.right is None: return current_depth
        left_depth = self.left.get_depth(current_depth + 1) if self.left else 0
        right_depth = self.right.get_depth(current_depth + 1) if self.right else 0
        return max(left_depth, right_depth)

    def collect_nodes_and_parents(self):
        nodes, queue = [], [(self, None)]
        while queue:
            current, parent = queue.pop(0)
            nodes.append((current, parent))
            if current.left: queue.append((current.left, current))
            if current.right: queue.append((current.right, current))
        return nodes

    def to_dict(self, id_counter=iter(range(1, 10000))):
        label = self._get_node_label()
        node_dict = {"name": label, "id": next(id_counter), "children": []}
        if self.left:
            node_dict["children"].append(self.left.to_dict(id_counter))
        if self.right:
            node_dict["children"].append(self.right.to_dict(id_counter))
        return node_dict

    def _get_node_label(self):
        if callable(self.data):
            return [name for name, func in FUNCTIONS.items() if func == self.data][0]
        elif isinstance(self.data, float):
            return f"{self.data:.3f}"
        return str(self.data)

    def __str__(self):
        if self.left is None and self.right is None:
            return self._get_node_label()
        return f"({self._get_node_label()} {str(self.left)} {str(self.right)})"

class GeneticProgramming:
    """ Encapsulates the entire GP evolutionary process. """
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
        dataset = []
        for x in range(-1, 6):
            for y in range(-1, 6):
                if x == -1: res = (2 * y - 17) / 3.0
                else: res = (y + 6 * x - 12) / 3.0
                dataset.append(((float(x), float(y)), res))
        return dataset

    def _initialize_population(self):
        logging.info("Initializing population using ramped half-and-half...")
        pop = []
        depth_range = range(self.params["min_initial_depth"], self.params["max_initial_depth"] + 1)
        inds_per_depth_method = self.params["population_size"] // (len(depth_range) * 2) if depth_range else 0
        for depth in depth_range:
            for _ in range(inds_per_depth_method):
                for method in ["full", "grow"]:
                    if len(pop) < self.params["population_size"]:
                        tree = GPTree()
                        tree.build_random_tree(0, depth, method=method)
                        pop.append(tree)
        while len(pop) < self.params["population_size"]:
            tree = GPTree()
            tree.build_random_tree(0, random.choice(depth_range), method=random.choice(["full", "grow"]))
            pop.append(tree)
        self.population = pop
        logging.info(f"Initialized population with {len(self.population)} individuals.")

    def _calculate_fitness(self, individual):
        errors = []
        for (x, y), target in self.dataset:
            prediction = individual.evaluate(x, y)
            if not isinstance(prediction, (int, float)) or math.isnan(prediction) or math.isinf(prediction) or abs(prediction) > 1e9:
                errors.append(1e12)
            else:
                errors.append((prediction - target)**2)
        return sum(errors) / len(errors)

    def _tournament_selection(self, population_with_fitness):
        t_size = min(self.params["tournament_size"], len(population_with_fitness))
        selected_contestants = random.sample(population_with_fitness, t_size)
        return min(selected_contestants, key=lambda item: item[0])[1]

    def _subtree_crossover(self, parent1, parent2):
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        nodes1 = [node for node, _ in child1.collect_nodes_and_parents()]
        nodes2 = [node for node, _ in child2.collect_nodes_and_parents()]
        if not nodes1 or not nodes2: return parent1, parent2
        graft1, graft2 = copy.deepcopy(random.choice(nodes2)), copy.deepcopy(random.choice(nodes1))
        self._replace_random_subtree(child1, graft1)
        self._replace_random_subtree(child2, graft2)
        if child1.get_depth() > self.params["max_tree_depth"]: child1 = parent1
        if child2.get_depth() > self.params["max_tree_depth"]: child2 = parent2
        return child1, child2

    def _replace_random_subtree(self, tree, graft):
        nodes_with_parents = tree.collect_nodes_and_parents()
        if not nodes_with_parents: return
        node_to_replace, parent = random.choice(nodes_with_parents)
        if parent is None:
            tree.data, tree.left, tree.right = graft.data, graft.left, graft.right
        elif parent.left == node_to_replace:
            parent.left = graft
        else:
            parent.right = graft

    def _point_mutation(self, individual):
        mutant = copy.deepcopy(individual)
        nodes = [node for node, _ in mutant.collect_nodes_and_parents()]
        if not nodes: return individual
        node_to_mutate = random.choice(nodes)
        if callable(node_to_mutate.data):
            node_to_mutate.data = random.choice([f for f in self.functions.values() if f != node_to_mutate.data])
        else:
            if isinstance(node_to_mutate.data, (int, float)) and random.random() < 0.5:
                perturbation = random.uniform(-0.5, 0.5)
                node_to_mutate.data = round(node_to_mutate.data + perturbation, 3)
            else:
                new_terminal = random.choice(self.terminals)
                if new_terminal == 'const':
                     node_to_mutate.data = round(random.uniform(*self.params["constant_range"]), 3)
                else:
                    node_to_mutate.data = new_terminal
        return mutant

    def run(self):
        logging.info(f"STARTING NEW GP RUN at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Parameters: {json.dumps(self.params, indent=2)}")
        self._initialize_population()
        for gen in range(1, self.params["generations"] + 1):
            population_with_fitness = sorted([(self._calculate_fitness(ind), ind) for ind in self.population], key=lambda item: item[0])
            current_best_fitness = population_with_fitness[0][0]
            self.fitness_history.append(current_best_fitness)
            if current_best_fitness < self.best_overall_fitness:
                self.best_overall_fitness = current_best_fitness
                self.best_overall_individual = copy.deepcopy(population_with_fitness[0][1])
                logging.info(f"Generation {gen:03d}: New best! MSE = {self.best_overall_fitness:.6f}, Nodes = {self.best_overall_individual.get_node_count()}")
            if self.best_overall_fitness <= self.params["desired_fitness_threshold"]:
                logging.info(f"Desired fitness threshold reached at generation {gen}.")
                break
            if gen == self.params["generations"]:
                logging.info("Maximum generations reached.")
                break
            new_population = [copy.deepcopy(ind) for _, ind in population_with_fitness[:self.params["elitism_count"]]]
            while len(new_population) < self.params["population_size"]:
                parent1, parent2 = self._tournament_selection(population_with_fitness), self._tournament_selection(population_with_fitness)
                if random.random() < self.params["crossover_rate"]:
                    offspring1, offspring2 = self._subtree_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
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
        if self.best_overall_individual:
            logging.info("\n--- Final Results ---")
            logging.info(f"Best Fitness (MSE): {self.best_overall_fitness:.8f}")
            logging.info(f"Best Individual (Prefix): {str(self.best_overall_individual)}")
            results_data = {
                "best_tree": self.best_overall_individual.to_dict(),
                "fitness_history": self.fitness_history,
                "final_mse": self.best_overall_fitness
            }
            try:
                with open("gp_results.json", "w") as f: json.dump(results_data, f, indent=4)
                logging.info("Results saved to gp_results.json for visualisation.")
            except Exception as e:
                logging.error(f"Could not save results to JSON file: {e}")
        else:
            logging.info("No solution was found.")

if __name__ == '__main__':
    start_time = time.time()
    gp_system = GeneticProgramming(GP_PARAMS, FUNCTIONS, TERMINALS)
    gp_system.run()
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")
