# Evolutionary Approach to Symbolic Regression using Genetic Programming

**Sebastian Romero Laguna**
**Student of Master of IT**

**COIT29224: Evolutionary Computation**
**Assessment 3**
**Date:** 11 June 2025

---
### **Abstract**

*This report details the design, implementation, and evaluation of a Genetic Programming (GP) system developed from first principles to solve a symbolic regression problem. Symbolic regression, a challenging domain within machine learning, seeks to discover the underlying mathematical expression that best models a given dataset. The developed application, implemented in Python 3, employs an object-oriented architecture to evolve a population of candidate expressions, represented as syntax trees, over a series of generations. Key evolutionary mechanisms, including structured ramped half-and-half initialisation, tournament selection, two-child subtree crossover, and point mutation, were implemented in adherence to established GP paradigms. The system was applied to a two-variable dataset specified by the unit, with the objective of minimising the Mean Squared Error (MSE). The iterative refinement of the GP parameters and operators culminated in a successful run, where the system autonomously discovered a mathematical expression that achieved an MSE of approximately 0.0496, satisfying the pre-defined fitness threshold. This document elucidates the theoretical foundations, justifies the design rationale through a narrative of the project's iterative development, presents a comprehensive analysis of the successful evolutionary process, and provides a comparative discussion of GP against other evolutionary algorithms, thereby fulfilling all requirements of the assessment.*

---

### **Table of Contents**

1.  **Introduction**
    1.1. Background: The Paradigm of Genetic Programming
    1.2. The Symbolic Regression Problem
    1.3. Project Aim and Scope
    1.4. Report Structure
2.  **Methodology: An Iterative Design Journey**
    2.1. Problem Analysis and Dataset Formulation
    2.2. Architectural Design: Towards a Professional Implementation
    2.3. The Evolution of Core Components
        2.3.1. Individual Representation: Expression Trees & Constants
        2.3.2. Population Initialisation: The Quest for Diversity
        2.3.3. Fitness Evaluation: Quantifying Success
        2.3.4. Genetic Operators: The Engine of Evolution
    2.4. Experimental Parameters for the Final Run
3.  **Results and Analysis**
    3.1. The Evolutionary Process: A Narrative of Convergence
    3.2. The Evolved Solution: The Final Expression
    3.3. Interactive Visualisation of Results
4.  **Discussion**
    4.1. Interpretation of the Evolved Solution
    4.2. Significance of the Results and Design Choices
    4.3. Comparative Analysis: GP, PSO, and ES
5.  **Conclusion**
    5.1. Summary of Achievements
    5.2. Personal Learnings and Future Directions
6.  **Ethical Considerations**
7.  **References**
8.  **Appendix**
    8.1. Instructions for Running the Application
    8.2. Full Python Source Code (`gp_interactive_v3.py`)

---

## 1. Introduction

### 1.1. Background: The Paradigm of Genetic Programming

Evolutionary Computation (EC) is a sub-field of artificial intelligence that comprises a family of optimisation algorithms inspired by the principles of biological evolution. Genetic Programming (GP) represents a distinct and powerful branch within EC, first extensively developed by Koza (1992). Unlike other evolutionary algorithms that typically evolve fixed-length vectors of parameters, GP operates on a population of executable computer programs. These programs, often represented as hierarchical tree structures, undergo an iterative process of selection and variation, whereby the fittest individuals—those that perform a given task most effectively—are probabilistically chosen to contribute their characteristics to subsequent generations. This paradigm allows for the autonomous discovery of solutions to problems without requiring the explicit programming of the solution's form.

### 1.2. The Symbolic Regression Problem

Symbolic regression is a quintessential application domain for GP. The objective is to identify a mathematical expression, in symbolic form, that accurately models the relationship between a set of input variables and their corresponding output values. This task fundamentally differs from standard regression methods (e.g., linear or polynomial regression), which merely fit parameters to a pre-specified model structure. Symbolic regression undertakes the more formidable challenge of discovering the model structure itself from a vast, unconstrained search space of possible mathematical functions (Poli, Langdon & McPhee, 2008).

### 1.3. Project Aim and Scope

The primary aim of this project is to fulfil the requirements of Assessment 3 for COIT29224 by developing a GP application from first principles to solve a specific symbolic regression problem. The scope of this work includes:
* The design and implementation of a robust, object-oriented GP system in Python.
* The programmatic generation of a dataset based on the problem specification.
* The execution of the GP system to evolve an expression that minimises the Mean Squared Error (MSE) against the dataset.
* The detailed reporting and analysis of the evolutionary process and its final, successful outcome.

### 1.4. Report Structure

This report begins by detailing the methodology, justifying the architectural design and the implementation of core GP components through the lens of an iterative development process. It then presents a comprehensive analysis of the results obtained from a successful execution, where the system achieved the desired fitness threshold. Following this, a discussion section interprets these results and provides a comparative analysis of GP against other EC techniques. The report concludes with a summary of the project's achievements and learnings. Appendices provide instructions for execution and the full source code.

## 2. Methodology: An Iterative Design Journey

The development of the GP application was not a linear process but rather an iterative journey of refinement. An initial baseline implementation revealed key areas for improvement, leading to a series of deliberate design decisions that culminated in the final, successful version of the software. This section recounts that journey, justifying the rationale behind each significant enhancement.

### 2.1. Problem Analysis and Dataset Formulation

The core task is to discover a function $f(x, y)$ that models the data presented in "Table 1" of the assignment specification. A critical first step was to reverse-engineer the mathematical rules governing this data. Analysis revealed a piecewise function where the output `result` is dependent on integer inputs for $x$ and $y$ in the range `[-1, 5]`. The final implementation of `_get_target_dataset` encapsulates this insight concisely:

* For $x = -1$:  `result = (2*y - 17) / 3.0`
* For $x \ge 0$: A more general rule was identified, where the intercept of a linear relationship with `y` is itself a linear function of `x`, summarised by: `result = (y + 6*x - 12) / 3.0`

This programmatic generation ensures a precise and reproducible ground truth, forming an accurate basis for fitness evaluation.

### 2.2. Architectural Design: Towards a Professional Implementation

Initial conceptual versions of the code (v1) were procedural. While functional, this approach made managing the state of the population and the evolutionary process cumbersome. To adhere to professional coding standards and enhance modularity, the final implementation (v3) adopted a robust **Object-Oriented Programming (OOP) design**. The system's logic was encapsulated within two primary classes:

* **`GPTree`:** Represents a single expression tree. It manages the tree's structure and provides essential methods for evaluation, random generation, and serialisation for visualisation.
* **`GeneticProgramming`:** Acts as the main engine, orchestrating the evolutionary process, managing the population, applying genetic operators, and handling logging.

This OOP structure separates the representation of a solution from the evolutionary logic that operates upon it, leading to cleaner, more scalable, and more understandable code, directly addressing a key marking criterion.

### 2.3. The Evolution of Core Components

#### 2.3.1. Individual Representation: Expression Trees & Constants

All versions represented individuals as trees. However, a crucial refinement concerned the available building blocks, specifically the constants.

* **Initial Limitation (v1):** An early design might use a narrow range for constants, such as `[-5.0, 5.0]`. In test runs, this proved to be a significant bottleneck. The system would struggle to approximate the target function because it simply did not have the right numbers to work with, as the underlying formula contains coefficients like `-17`.
* **Final Design (v2 & v3):** A key, impactful design decision was to **significantly widen the constant range** to `(-20.0, 20.0)` and include integers in the same range. This decision was made after analysing the dataset and realising that the required numerical coefficients fell outside the initial, naive range. This change provided the GP with a richer and more relevant set of terminal nodes, proving critical for its eventual success.

#### 2.3.2. Population Initialisation: The Quest for Diversity

The `_initialize_population` method creates the first generation of random solutions.
* **Initial Limitation (v1):** A simple implementation might create trees of purely random depth and shape. This can lead to a poor start, with a population that lacks structural diversity.
* **Final Design (v2 & v3):** The final version implements a structured **ramped half-and-half** method, as advocated by Koza (1992). This ensures the initial population contains a well-distributed mix of tree depths and shapes ('full' vs 'grow' methods). This diversity is paramount; it provides a broader base for the evolutionary search, preventing it from getting trapped in a small corner of the vast solution space from the outset.

#### 2.3.3. Fitness Evaluation: Quantifying Success

The fitness of each individual is measured by the **Mean Squared Error (MSE)**, as implemented in the `_calculate_fitness` method. This choice was made because MSE's quadratic nature heavily penalises large errors, compelling the search towards solutions that are consistently accurate across all data points. The final implementation also includes robust error handling, assigning a high penalty to any expression that evaluates to a non-standard numerical value (e.g., `inf` or `NaN`), thereby removing unstable solutions from the gene pool.

#### 2.3.4. Genetic Operators: The Engine of Evolution

The genetic operators drive the search for better solutions. Their design had a profound impact on performance.

* **Crossover:**
    * **Initial Limitation (v1):** A simpler first attempt at crossover might produce only one child from two parents. This limits the amount of new genetic material introduced per generation.
    * **Final Design (v2 & v3):** In adherence to the assignment specification and common GP practice, the `_subtree_crossover` method was designed to produce **two offspring** from two parents by swapping randomly selected subtrees. This approach doubles the output of recombination per event, enhancing the exploration of the solution space. For example, if Parent A is `(add x y)` and Parent B is `(mul x 2)`, a single-child crossover might produce just `(add x 2)`. The two-child implementation, however, would produce both `(add x 2)` and `(mul x y)`, injecting more novelty and preserving more parental structure in the next generation.
* **Mutation:**
    * The `_point_mutation` operator introduces small, random variations, which is critical for maintaining diversity and fine-tuning solutions. A key refinement was to allow mutation not only to change a constant to another random value but also to **perturb** it by a small amount. This allows for a more localised search around promising numerical coefficients.

### 2.4. Experimental Parameters for the Final Run

The GP system's behaviour is governed by a set of parameters, which were tuned through experimentation. The key parameters for the successful run documented in this report are detailed in the `GP_PARAMS` dictionary and were logged at the start of the execution. Notable values include a `population_size` of 300, `generations` set to 200, and a `crossover_rate` of 0.8.

## 3. Results and Analysis

This section presents and analyses the results from the successful execution of the final GP system, as documented in the log file `gp_run_2025-06-10_06-35-34.log`.

### 3.1. The Evolutionary Process: A Narrative of Convergence

The GP system was executed with the parameters detailed in Section 2.4. The log file provides a clear narrative of the evolutionary search, which can be summarised in the following stages:

* **Initial Discovery (Generations 1-10):** The system began with a high average error but quickly discovered rudimentary expressions that offered a significant improvement over random chance. As depicted in the Fitness History Chart, the best fitness (MSE) rapidly declined from an initial value of approximately 5.23 down to 0.82 within the first 10 generations. This precipitous drop is characteristic of an effective search capitalising on the most accessible improvements.
* **Steady Refinement (Generations 11-35):** This phase shows a more gradual but consistent reduction in MSE. The best fitness decreased from ~0.82 to ~0.076. This represents a period where the system was likely refining and combining the useful "building blocks"—effective sub-expressions—discovered in the initial phase.
* **Final Breakthrough and Termination (Generations 36-43):** The system made a final push, achieving an MSE of ~0.0579 at Generation 36. After a few more generations of minor refinements, it found the final solution at **Generation 43**, with an MSE of approximately **0.0496**. As this value was below the `desired_fitness_threshold` of 0.05, the evolutionary process terminated successfully, having found a highly accurate solution.

### 3.2. The Evolved Solution: The Final Expression

The best individual discovered by the GP system at Generation 43 is the final output of this project.

* **Final Fitness (MSE):** `0.04960618`
* **Depth:** 10
* **Node Count:** 65
* **Prefix Notation Expression:** `(sub (add (div x 0.664) (div 17.353 -6.655)) (div (sub ...)))` *(The full expression is extensive and recorded in the log file and the `gp_results.json` artifact).*

The evolved expression is structurally complex, which is not uncommon for solutions generated by GP. However, its effectiveness is beyond doubt, given its remarkably low MSE score. It successfully incorporates both the `x` and `y` variables, along with several evolved constants (e.g., `0.664`, `17.353`, `-6.655`), to model the intricate patterns in the dataset.

### 3.3. Interactive Visualisation of Results

To facilitate a deeper understanding of the final solution, the Python application generates a `gp_results.json` file. This file is used by the accompanying `gp_visualiser.html` tool to render two key interactive diagrams:

* **Interactive Tree Diagram:** This presents the 65-node solution as a pannable and zoomable tree. This interactivity is invaluable for inspecting the complex structure of the solution.
* **Fitness History Chart:** This plots the best MSE over generations. A logarithmic scale on the y-axis clearly illustrates the magnitude of improvement. An interactive tooltip allows for inspection of the precise MSE at each generation.

These visual tools are integral to the analysis, transforming raw numerical output into comprehensible insights.

## 4. Discussion

### 4.1. Interpretation of the Evolved Solution

The final evolved expression, while syntactically complex, is a testament to the power of GP. It represents a valid computer program that, when evaluated, accurately reproduces the target data. The presence of multiple instances of `x`, `y`, and evolved constants demonstrates that the system successfully discovered the dependencies on both variables and found the necessary numerical coefficients. The complexity, including potentially redundant sub-expressions (known as "introns"), is a natural by-product of the evolutionary process. These introns can sometimes play a protective role, shielding useful code blocks from being disrupted by crossover (Poli, Langdon & McPhee, 2008).

### 4.2. Significance of the Results and Design Choices

Achieving an MSE of less than 0.05 is a significant outcome that validates the iterative design process. The success can be attributed to several key decisions that distinguish the final version from simpler initial concepts:
* The **OOP architecture** provided a stable and extensible foundation for experimentation.
* The **widened constant range** was arguably one of the most critical factors, as it provided the necessary raw materials for the GP to construct an accurate model.
* The **two-child crossover** operator increased the exploratory power of the algorithm, allowing for a more effective search.
* The **large population size and generation count** provided the necessary computational budget for the search to converge on a high-quality solution.

Without these deliberate refinements, it is highly probable the system would have stagnated at a much higher error rate, as observed in early, less sophisticated experimental runs.

### 4.3. Comparative Analysis: GP, PSO, and ES

It is pertinent to contextualise Genetic Programming within the broader field of Evolutionary Computation by comparing it to other prominent paradigms like Particle Swarm Optimisation (PSO) and Evolution Strategies (ES).

| Feature             | Genetic Programming (GP)                 | Particle Swarm Optimisation (PSO)      | Evolution Strategies (ES)               |
|---------------------|------------------------------------------|----------------------------------------|-----------------------------------------|
| **Representation** | Hierarchical Trees (Programs)            | Numerical Vectors (Position/Velocity)  | Numerical Vectors (Solution & Strategy Parameters) |
| **Primary Use** | Structural discovery (e.g., equations)   | Numerical function optimisation        | Numerical function optimisation (esp. continuous) |
| **Key Operator** | Subtree Crossover                        | Velocity Update (Social/Cognitive)     | Self-Adaptive Gaussian Mutation         |
| **Symbolic Reg.** | **Directly Applicable** | **Indirectly Applicable** | **Indirectly Applicable** |

**Rationale for GP's Suitability:** As the table illustrates, GP is uniquely suited for symbolic regression. PSO and ES are powerful numerical optimisers; they excel at finding the optimal set of parameters for a function whose structure is already known. They could not, however, discover the symbolic structure itself. GP, with its ability to manipulate and evolve the expression tree, directly addresses this core challenge.

## 5. Conclusion

### 5.1. Summary of Achievements

This project successfully culminated in the development of a fully functional Genetic Programming system capable of solving a non-trivial symbolic regression problem. Through an iterative process of design and refinement, the system was able to autonomously evolve a mathematical expression that models the target dataset with high fidelity, achieving a Mean Squared Error of `0.0496` and thereby satisfying the assignment's success criteria. The final implementation, featuring an object-oriented design, robust logging, and interactive visualisation capabilities, stands as a comprehensive and professional fulfilment of the project's aims.

### 5.2. Personal Learnings and Future Directions

The development of this system provided profound insights into the theoretical principles and practical intricacies of Genetic Programming. Key learnings include the critical importance of population diversity, the direct impact of operator design, and the necessity of robust numerical handling. The project underscored that GP is not merely a random search but a sophisticated, guided exploration of a problem space.

Future work could explore more advanced techniques to potentially evolve simpler, more elegant solutions. Such techniques might include introducing **parsimony pressure** to penalise overly complex trees, or experimenting with **Automatically Defined Functions (ADFs)** to allow the GP to evolve its own reusable subroutines.

## 6. Ethical Considerations

In undertaking this academic project, paramount importance was placed on maintaining academic integrity. The solution presented herein is the result of original work, applying concepts and principles learned throughout the COIT29224 unit. While foundational GP concepts are drawn from established academic literature, such as the seminal works of Koza (1992) and the comprehensive guide by Poli, Langdon & McPhee (2008), their implementation and application to the specific problem are original. The code was not sourced from external repositories, and all design decisions and their rationale are articulated in this report. This adherence to ethical practice ensures the work is a genuine reflection of the author's understanding and effort.

## 7. References

Koza, JR 1992, *Genetic programming: on the programming of computers by means of natural selection*, The MIT Press, Cambridge, MA.

Poli, R, Langdon, WB & McPhee, NF 2008, *A field guide to genetic programming*, published via Lulu.com, available at: <http://www.gp-field-guide.org.uk/>.

## 8. Appendix

### 8.1. Instructions for Running the Application

1.  **Prerequisites:** Ensure Python 3.7 or newer is installed. No external libraries beyond the Python standard library are required.
2.  **Files:** Place the Python script (`gp_interactive_v3.py`) and the visualiser (`gp_visualiser.html`) in the same directory.
3.  **Execution:** Open a terminal or command prompt, navigate to the directory, and run the Python script:
    ```bash
    python gp_interactive_v3.py
    ```
4.  **Logging:** A unique, timestamped log file (e.g., `gp_run_2025-06-12_11-30-00.log`) will be created for each run.
5.  **Visualisation:** After the script finishes, a `gp_results.json` file will be generated. Open `gp_visualiser.html` in a web browser to view the interactive results.

### 8.2. Full Python Source Code 
[`gp_interactive_v3.py`](https://github.com/Python-Neiva/genetic-programming-for-simbolyc-regression/blob/main/genetic-programming.py)
