#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Programming for Symbolic Regression (Version 2)
Task: Evolve a mathematical expression f(x, y) that fits a given dataset.
Based on COIT29224 Assignment 3 Specifications.
Key improvements in v2:
- Crossover produces two children.
- More structured ramped half-and-half initialization.
- Adjusted GP parameters for potentially better exploration (population, generations, constant ranges).
- Refined ASCII tree printing.
"""

import random
import math
import operator
import copy
import time

# ── GP Parameters (v2 Tuned) ──────────────────────────────────────────────────
POPULATION_SIZE   = 300    # Increased population size
MIN_INITIAL_DEPTH = 2      # Minimum depth of trees at initial generation.
MAX_INITIAL_DEPTH = 4      # Slightly reduced max initial depth to encourage smaller initial solutions
MAX_TREE_DEPTH    = 10     # Overall maximum depth for any tree (increased slightly)
GENERATIONS       = 200    # Increased generations
TOURNAMENT_SIZE   = 5      # Size of the tournament for parent selection.
CROSSOVER_RATE    = 0.8    # Probability that crossover will occur.
MUTATION_RATE     = 0.25   # Probability that an individual (offspring) will undergo mutation.

DESIRED_FITNESS_THRESHOLD = 0.05 # Made target slightly more ambitious if 0.1 is reached easily
EPSILON = 1e-6

# ── Primitives: Functions and Terminals ─────────────────────────────────────
def safe_div(x, y):
    if abs(y) < EPSILON:
        return 1.0 # Return a defined value for division by zero
    return x / y

FUNCTIONS = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': safe_div
}

TERMINALS = ['x', 'y', 'const']
# Widened constant ranges based on dataset values (e.g., -17 appears in underlying formula)
CONSTANT_RANGE = (-20.0, 20.0)
INTEGER_CONSTANTS = list(range(-20, 21))

# ── Dataset ─────────────────────────────────────────────────────────────────
def get_target_dataset():
    """
    Generates the target dataset (list of ((x,y), result) tuples).
    The data is derived from Table 1 in the assignment specification.
    x and y range from -1 to 5.
    """
    dataset = []
    x_values = range(-1, 6)
    y_values = range(-1, 6)

    for x_val in x_values:
        for y_val in y_values:
            if x_val == -1: result = (2 * y_val - 17) / 3.0
            elif x_val == 0: result = (y_val - 12) / 3.0
            elif x_val == 1: result = (y_val - 6) / 3.0
            elif x_val == 2: result = y_val / 3.0
            elif x_val == 3: result = (y_val + 6) / 3.0
            elif x_val == 4: result = (y_val + 12) / 3.0
            elif x_val == 5: result = (y_val + 18) / 3.0
            else: result = 0
            dataset.append(((float(x_val), float(y_val)), result))
    return dataset

# ── Expression Tree Structure (GPTree) ──────────────────────────────────────
class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_terminal(self):
        return self.left is None and self.right is None

    def evaluate(self, x_val, y_val):
        if callable(self.data):
            if self.left is None or self.right is None:
                # This case should ideally not be reached if trees are well-formed.
                # It implies a function node is missing one or both children.
                # print("Warning: Malformed tree - function node missing children during evaluation.")
                return 1e9 # Return a large penalty value

            left_result = self.left.evaluate(x_val, y_val)
            right_result = self.right.evaluate(x_val, y_val)

            # Check for problematic intermediate results before applying the function
            if math.isinf(left_result) or math.isnan(left_result) or \
               math.isinf(right_result) or math.isnan(right_result):
                return 1e9 # Penalize if children evaluate to inf/nan

            try:
                result = self.data(left_result, right_result)
                if math.isinf(result) or math.isnan(result):
                    return 1e9 # Penalize if operation results in inf/nan
                return result
            except (OverflowError, ValueError, ZeroDivisionError): # ZeroDivisionError should be caught by safe_div for FUNCTION['div']
                return 1e9 # Return a large penalty value for numerical issues
        elif self.data == 'x':
            return x_val
        elif self.data == 'y':
            return y_val
        else: # Constant
            try:
                return float(self.data)
            except ValueError: # Should not happen if constants are always numbers
                # print(f"Warning: Constant {self.data} is not a float.")
                return 0.0

    def get_depth(self, current_depth=0):
        if self.is_terminal():
            return current_depth

        left_depth = self.left.get_depth(current_depth + 1) if self.left else current_depth
        right_depth = self.right.get_depth(current_depth + 1) if self.right else current_depth

        return max(left_depth, right_depth)

    def build_random_tree(self, current_depth, max_depth, method="grow"):
        if current_depth >= max_depth or \
           (method == "grow" and random.random() < 0.5 and current_depth >= MIN_INITIAL_DEPTH): # Increased chance for terminals in grow
            chosen_terminal = random.choice(TERMINALS)
            if chosen_terminal == 'const':
                if random.random() < 0.7: # More floats
                    self.data = round(random.uniform(CONSTANT_RANGE[0], CONSTANT_RANGE[1]), 3)
                else:
                    self.data = float(random.choice(INTEGER_CONSTANTS))
            else:
                self.data = chosen_terminal
            self.left = None
            self.right = None
        else:
            func_name = random.choice(list(FUNCTIONS.keys()))
            self.data = FUNCTIONS[func_name]

            self.left = GPTree()
            self.left.build_random_tree(current_depth + 1, max_depth, method)

            self.right = GPTree()
            self.right.build_random_tree(current_depth + 1, max_depth, method)

    def __str__(self):
        if callable(self.data):
            try:
                op_name = [k for k, v in FUNCTIONS.items() if v == self.data][0]
            except IndexError: # Should not happen if self.data is always from FUNCTIONS
                op_name = "unknown_op"
            return f"({op_name} {str(self.left)} {str(self.right)})"
        else:
            if isinstance(self.data, float):
                return f"{self.data:.3f}"
            return str(self.data)

    def copy(self):
        return copy.deepcopy(self)

    def collect_nodes_and_parents(self):
        """ Collects all nodes. Returns a list of (node, parent_node) tuples. Root's parent is None. """
        nodes = []
        queue = [(self, None)] # (node, parent)
        while queue:
            current_node, parent = queue.pop(0)
            nodes.append((current_node, parent))
            if current_node.left:
                queue.append((current_node.left, current_node))
            if current_node.right:
                queue.append((current_node.right, current_node))
        return nodes

    def get_node_count(self):
        if self.is_terminal():
            return 1
        count = 1
        if self.left: count += self.left.get_node_count()
        if self.right: count += self.right.get_node_count()
        return count

    def _node_label(self):
        if callable(self.data):
            return [k for k, v in FUNCTIONS.items() if v == self.data][0]
        elif isinstance(self.data, float):
            return f"{self.data:.2f}"
        return str(self.data)

    def print_tree_ascii(self, indent_str="", last_child=True):
        """ Improved ASCII tree printing """
        branch = "`-- " if last_child else "|-- "
        print(f"{indent_str}{branch}{self._node_label()}")

        indent_str += "    " if last_child else "|   "

        children = []
        if self.left: children.append(self.left)
        if self.right: children.append(self.right)

        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            child.print_tree_ascii(indent_str, is_last)


# ── GP Core Algorithm Functions ─────────────────────────────────────────────

def initialize_population(pop_size, min_init_depth, max_init_depth):
    """
    Creates the initial population using a more structured ramped half-and-half method.
    """
    population = []
    num_depth_levels = max_init_depth - min_init_depth + 1
    if num_depth_levels <= 0: num_depth_levels = 1 # Should not happen with valid depths

    target_inds_per_depth = pop_size // num_depth_levels

    current_inds = 0
    for depth_level in range(num_depth_levels):
        current_max_depth = min_init_depth + depth_level

        # Determine how many individuals for this depth, distribute remainder
        inds_for_this_depth = target_inds_per_depth
        if depth_level == num_depth_levels -1 : # last depth level gets remainder
            inds_for_this_depth = pop_size - current_inds

        inds_per_method = inds_for_this_depth // 2

        for _ in range(inds_per_method): # Grow method
            if current_inds < pop_size:
                ind = GPTree()
                ind.build_random_tree(0, current_max_depth, "grow")
                population.append(ind)
                current_inds +=1

        for _ in range(inds_for_this_depth - inds_per_method): # Full method (takes remainder for this depth)
             if current_inds < pop_size:
                ind = GPTree()
                ind.build_random_tree(0, current_max_depth, "full")
                population.append(ind)
                current_inds +=1

    # If pop_size was not perfectly divisible, fill remaining spots
    while len(population) < pop_size:
        depth = random.randint(min_init_depth, max_init_depth)
        method = "grow" if random.random() < 0.5 else "full"
        ind = GPTree()
        ind.build_random_tree(0, depth, method)
        population.append(ind)

    return population


def calculate_fitness(individual, dataset):
    if not dataset: return float('inf')
    squared_errors = []
    for (x_val, y_val), target_result in dataset:
        predicted_result = individual.evaluate(x_val, y_val)
        if isinstance(predicted_result, (int, float)) and \
           not math.isnan(predicted_result) and not math.isinf(predicted_result) and \
           abs(predicted_result) < 1e10: # Check for extremely large values as well
            error = predicted_result - target_result
            squared_errors.append(error * error)
        else:
            # Penalize individuals that produce non-numeric, inf, nan or extremely large results
            squared_errors.append(1e12) # Very large error penalty

    if not squared_errors: return float('inf')
    return sum(squared_errors) / len(squared_errors)

def tournament_selection(population_with_fitness, t_size):
    if not population_with_fitness: return None
    actual_t_size = min(t_size, len(population_with_fitness))
    if actual_t_size == 0: return None

    selected_contestants = random.sample(population_with_fitness, actual_t_size)
    selected_contestants.sort(key=lambda item: item[0]) # Lower fitness is better
    return selected_contestants[0][1]

def subtree_crossover(parent1_orig, parent2_orig, max_depth_limit):
    """
    Performs subtree crossover, creating two children by swapping subtrees.
    Ensures children do not exceed max_depth_limit.
    """
    p1 = parent1_orig.copy()
    p2 = parent2_orig.copy()

    # Collect nodes for parent1
    nodes_p1 = [node for node, _ in p1.collect_nodes_and_parents()]
    if not nodes_p1: return parent1_orig.copy(), parent2_orig.copy() # Should not happen

    # Collect nodes for parent2
    nodes_p2 = [node for node, _ in p2.collect_nodes_and_parents()]
    if not nodes_p2: return parent1_orig.copy(), parent2_orig.copy()

    # Crossover for child 1 (p1 gets subtree from p2)
    # Choose crossover point in p1 (node_in_p1_to_replace)
    # Choose subtree in p2 (subtree_from_p2)

    # It's safer to choose any node, not just non-terminals, for replacement.
    # If a terminal is chosen in p1, and a non-terminal subtree from p2 is grafted, p1 grows.
    # If a non-terminal in p1 is chosen, and a terminal from p2 is grafted, p1 shrinks.

    # Child 1: p1_copy takes subtree from p2_copy
    child1 = parent1_orig.copy()
    child1_nodes_with_parents = child1.collect_nodes_and_parents()
    if not child1_nodes_with_parents: return parent1_orig.copy(), parent2_orig.copy()

    # Pick random node in child1 to be replaced
    idx1 = random.randrange(len(child1_nodes_with_parents))
    node_to_replace_in_child1, parent_of_node_in_child1 = child1_nodes_with_parents[idx1]

    # Pick random subtree from parent2
    parent2_nodes = [node for node, _ in parent2_orig.collect_nodes_and_parents()]
    if not parent2_nodes: return parent1_orig.copy(), parent2_orig.copy()
    graft_subtree_from_p2 = random.choice(parent2_nodes).copy()

    if parent_of_node_in_child1 is None: # Replacing root of child1
        child1 = graft_subtree_from_p2
    elif parent_of_node_in_child1.left == node_to_replace_in_child1:
        parent_of_node_in_child1.left = graft_subtree_from_p2
    else: # Must be right child
        parent_of_node_in_child1.right = graft_subtree_from_p2

    if child1.get_depth() > max_depth_limit:
        child1 = parent1_orig.copy() # Revert if too deep

    # Child 2: p2_copy takes subtree from p1_copy
    child2 = parent2_orig.copy()
    child2_nodes_with_parents = child2.collect_nodes_and_parents()
    if not child2_nodes_with_parents: return child1, parent2_orig.copy() # child1 might be ok

    idx2 = random.randrange(len(child2_nodes_with_parents))
    node_to_replace_in_child2, parent_of_node_in_child2 = child2_nodes_with_parents[idx2]

    parent1_nodes = [node for node, _ in parent1_orig.collect_nodes_and_parents()]
    if not parent1_nodes: return child1, parent2_orig.copy()
    graft_subtree_from_p1 = random.choice(parent1_nodes).copy()

    if parent_of_node_in_child2 is None: # Replacing root of child2
        child2 = graft_subtree_from_p1
    elif parent_of_node_in_child2.left == node_to_replace_in_child2:
        parent_of_node_in_child2.left = graft_subtree_from_p1
    else:
        parent_of_node_in_child2.right = graft_subtree_from_p1

    if child2.get_depth() > max_depth_limit:
        child2 = parent2_orig.copy() # Revert if too deep

    return child1, child2


def point_mutation(individual_orig, max_depth_limit):
    """
    Performs point mutation on an individual. One random node is selected and its content is changed.
    - Function node: changes to another function.
    - Terminal node (variable): changes to another variable or a constant.
    - Terminal node (constant): changes to another constant or another variable.
    Ensures mutated tree does not exceed max_depth_limit (less critical for point mutation).
    """
    individual = individual_orig.copy()

    nodes_info = individual.collect_nodes_and_parents()
    if not nodes_info: return individual_orig.copy()

    node_to_mutate, _ = random.choice(nodes_info) # Parent info not strictly needed for point mutation logic here

    if callable(node_to_mutate.data): # Function node
        current_func_name = [k for k,v in FUNCTIONS.items() if v == node_to_mutate.data][0]
        possible_new_funcs = [f_name for f_name in FUNCTIONS.keys() if f_name != current_func_name]
        if possible_new_funcs:
            new_func_name = random.choice(possible_new_funcs)
            node_to_mutate.data = FUNCTIONS[new_func_name]
        # Children (left, right) remain, arity is preserved as all are binary.
    else: # Terminal node
        # Change to any other terminal type ('x', 'y', or a new 'const')
        # Or, if it's a const, it could also change its value.

        # Option 1: Change type or value
        new_terminal_choice = random.choice(TERMINALS + ['modify_const_value_if_const'])

        if node_to_mutate.data == 'x' or node_to_mutate.data == 'y' or new_terminal_choice != 'modify_const_value_if_const':
            # Change to a new terminal type
            new_terminal_type = random.choice(TERMINALS)
            if new_terminal_type == 'x': node_to_mutate.data = 'x'
            elif new_terminal_type == 'y': node_to_mutate.data = 'y'
            else: # 'const'
                if random.random() < 0.7:
                    node_to_mutate.data = round(random.uniform(CONSTANT_RANGE[0], CONSTANT_RANGE[1]), 3)
                else:
                    node_to_mutate.data = float(random.choice(INTEGER_CONSTANTS))
        else: # It's a constant and we chose to modify its value
             if isinstance(node_to_mutate.data, (int, float)): # Check if it's already a number
                # Small perturbation or full re-randomization
                if random.random() < 0.5: # Perturb
                    perturbation = random.uniform(-1.0, 1.0) * (CONSTANT_RANGE[1] - CONSTANT_RANGE[0]) * 0.1 # 10% of range
                    new_val = node_to_mutate.data + perturbation
                    node_to_mutate.data = round(max(CONSTANT_RANGE[0], min(CONSTANT_RANGE[1], new_val)), 3)
                else: # Re-randomize
                    if random.random() < 0.7:
                        node_to_mutate.data = round(random.uniform(CONSTANT_RANGE[0], CONSTANT_RANGE[1]), 3)
                    else:
                        node_to_mutate.data = float(random.choice(INTEGER_CONSTANTS))
             else: # If it was 'const' but not yet a number (should not happen after init)
                if random.random() < 0.7:
                    node_to_mutate.data = round(random.uniform(CONSTANT_RANGE[0], CONSTANT_RANGE[1]), 3)
                else:
                    node_to_mutate.data = float(random.choice(INTEGER_CONSTANTS))


        # Ensure it remains a terminal (no children)
        node_to_mutate.left = None
        node_to_mutate.right = None

    # Depth check (usually not an issue for point mutation unless a terminal became a function, which is not the case here)
    if individual.get_depth() > max_depth_limit:
         return individual_orig.copy()

    return individual

# ── Main Evolutionary Loop ──────────────────────────────────────────────────
def run_gp_evolution():
    print("Starting Genetic Programming (v2) for Symbolic Regression...")
    start_time = time.time()

    dataset = get_target_dataset()
    if not dataset:
        print("Error: Dataset is empty.")
        return

    print(f"Dataset loaded with {len(dataset)} points.")

    population = initialize_population(POPULATION_SIZE, MIN_INITIAL_DEPTH, MAX_INITIAL_DEPTH)
    print(f"Initial population of {len(population)} individuals created using structured ramped half-and-half.")

    best_overall_individual = None
    best_overall_fitness = float('inf')

    # For tracking fitness progress for potential plotting
    fitness_history = []


    for gen in range(1, GENERATIONS + 1):
        gen_start_time = time.time()

        population_with_fitness = []
        for ind in population:
            fitness = calculate_fitness(ind, dataset)
            population_with_fitness.append((fitness, ind))

        population_with_fitness.sort(key=lambda item: item[0])

        current_best_fitness_in_gen = population_with_fitness[0][0]
        current_best_individual_in_gen = population_with_fitness[0][1]
        fitness_history.append(current_best_fitness_in_gen)

        if current_best_fitness_in_gen < best_overall_fitness:
            best_overall_fitness = current_best_fitness_in_gen
            best_overall_individual = current_best_individual_in_gen.copy()
            print(f"Generation {gen:03d}: New best! Fitness (MSE) = {best_overall_fitness:.6f}, Nodes = {best_overall_individual.get_node_count()}")
        else:
            avg_fitness_gen = sum(f for f, _ in population_with_fitness) / len(population_with_fitness)
            print(f"Generation {gen:03d}: Best this gen MSE = {current_best_fitness_in_gen:.6f} (Avg: {avg_fitness_gen:.6f}, Overall best: {best_overall_fitness:.6f})")

        if best_overall_fitness <= DESIRED_FITNESS_THRESHOLD:
            print(f"\nDesired fitness threshold ({DESIRED_FITNESS_THRESHOLD}) reached!")
            break

        if gen == GENERATIONS:
            print("\nMaximum generations reached.")
            break

        new_population = []

        # Elitism: Keep the top N individuals (e.g., 1 or 2)
        num_elites = max(1, int(0.02 * POPULATION_SIZE)) # Keep top 2% or at least 1
        for i in range(min(num_elites, len(population_with_fitness))):
             new_population.append(population_with_fitness[i][1].copy())

        # Fill the rest of the population using selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population_with_fitness, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population_with_fitness, TOURNAMENT_SIZE)

            if parent1 is None or parent2 is None: # Should not happen if pop_with_fitness is valid
                # Fallback: create random individuals if selection fails
                depth = random.randint(MIN_INITIAL_DEPTH, MAX_INITIAL_DEPTH)
                method = "grow" if random.random() < 0.5 else "full"
                temp_ind = GPTree()
                temp_ind.build_random_tree(0, depth, method)
                new_population.append(temp_ind)
                if len(new_population) >= POPULATION_SIZE: break
                continue

            offspring1, offspring2 = parent1.copy(), parent2.copy() # Default to copies

            if random.random() < CROSSOVER_RATE:
                offspring1, offspring2 = subtree_crossover(parent1, parent2, MAX_TREE_DEPTH)

            # Mutation
            if random.random() < MUTATION_RATE:
                offspring1 = point_mutation(offspring1, MAX_TREE_DEPTH)
            if random.random() < MUTATION_RATE: # Independent mutation chance for second offspring
                offspring2 = point_mutation(offspring2, MAX_TREE_DEPTH)

            new_population.append(offspring1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(offspring2)

        population = new_population[:POPULATION_SIZE]
        gen_time_taken = time.time() - gen_start_time
        # print(f"Generation {gen:03d} took {gen_time_taken:.2f}s. Population size: {len(population)}")


    end_time = time.time()
    total_time_taken = end_time - start_time
    print("\n=== Evolutionary Process Ended ===")
    print(f"Total time taken: {total_time_taken:.2f} seconds.")

    if best_overall_individual:
        print("\nBest individual found:")
        print(f"  Fitness (MSE): {best_overall_fitness:.8f}")
        print(f"  Depth: {best_overall_individual.get_depth()}")
        print(f"  Node count: {best_overall_individual.get_node_count()}")
        print(f"  Expression (prefix): {str(best_overall_individual)}")
        print("  Tree structure:")
        best_overall_individual.print_tree_ascii()

        print("\nSample evaluations of the best individual:")
        sample_points_eval = [((-1.0,-1.0),-19.0/3.0), ((0.0,0.0),-4.0), ((2.0,3.0),1.0), ((5.0,5.0),23.0/3.0), ((1.0, 1.0), (1-6)/3.0)]
        for (x,y), expected in sample_points_eval:
            predicted = best_overall_individual.evaluate(x,y)
            print(f"  f({x},{y}): Predicted={predicted:.4f}, Expected={expected:.4f}, Diff={abs(predicted-expected):.4f}")
    else:
        print("No solution found that improved from initial state.")



if __name__ == "__main__":
    # random.seed(42) # For reproducible runs during development
    run_gp_evolution()

