import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable

class GeneticAlgorithm:
    def __init__(
        self, 
        objective_function: Callable,
        num_variables: int,
        variable_bounds: List[Tuple[float, float]],
        population_size: int = 100,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
        use_adaptive_mutation: bool = True,
        selection_method: str = "tournament",
        crossover_method: str = "simulated_binary",
        mutation_method: str = "gaussian"
    ):
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.variable_bounds = variable_bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.use_adaptive_mutation = use_adaptive_mutation
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        
        self.best_fitness_history = []
        self.best_solution_history = []
        self.mean_fitness_history = []
        self.diversity_history = []

    def initialize_population(self) -> List[List[float]]:
        return [
            [random.uniform(bound[0], bound[1]) for bound in self.variable_bounds]
            for _ in range(self.population_size)
        ]

    def calculate_fitness(self, population: List[List[float]]) -> List[float]:
        return [self.objective_function(individual[0]) for individual in population]

    def select_parents(self, population: List[List[float]], fitness: List[float]) -> List[List[float]]:
        if self.selection_method == "tournament":
            return self.tournament_selection(population, fitness)
        elif self.selection_method == "roulette":
            return self.roulette_selection(population, fitness)
        elif self.selection_method == "rank":
            return self.rank_selection(population, fitness)
        return self.tournament_selection(population, fitness)

    def tournament_selection(self, population: List[List[float]], fitness: List[float], tournament_size: int = 3) -> List[List[float]]:
        selected_parents = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner])
        return selected_parents

    def roulette_selection(self, population: List[List[float]], fitness: List[float]) -> List[List[float]]:
        fitness = np.array(fitness)
        fitness = fitness - min(fitness)
        total_fitness = sum(fitness)
        if total_fitness == 0:
            return random.choices(population, k=self.population_size)
        probabilities = fitness / total_fitness
        return random.choices(population, weights=probabilities, k=self.population_size)

    def rank_selection(self, population: List[List[float]], fitness: List[float]) -> List[List[float]]:
        sorted_indices = np.argsort(fitness)
        ranks = np.arange(1, len(fitness) + 1)
        probabilities = ranks / sum(ranks)
        selected_indices = np.random.choice(
            len(population),
            size=self.population_size,
            p=probabilities,
            replace=True
        )
        return [population[i] for i in selected_indices]

    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        if self.crossover_method == "simulated_binary":
            return self.simulated_binary_crossover(parent1, parent2)
        elif self.crossover_method == "single_point":
            return self.single_point_crossover(parent1, parent2)
        elif self.crossover_method == "uniform":
            return self.uniform_crossover(parent1, parent2)
        return self.simulated_binary_crossover(parent1, parent2)

    def simulated_binary_crossover(self, parent1: List[float], parent2: List[float], eta: float = 15) -> Tuple[List[float], List[float]]:
        if random.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()

        u = random.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))

        child1 = [0.5 * ((1 + beta) * p1 + (1 - beta) * p2) for p1, p2 in zip(parent1, parent2)]
        child2 = [0.5 * ((1 - beta) * p1 + (1 + beta) * p2) for p1, p2 in zip(parent1, parent2)]

        child1 = [max(bound[0], min(val, bound[1])) for val, bound in zip(child1, self.variable_bounds)]
        child2 = [max(bound[0], min(val, bound[1])) for val, bound in zip(child2, self.variable_bounds)]

        return child1, child2

    def single_point_crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        if random.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def uniform_crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        if random.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = []
        child2 = []
        for p1, p2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(p1)
                child2.append(p2)
            else:
                child1.append(p2)
                child2.append(p1)
        return child1, child2

    def perform_mutation(self, individual: List[float], current_generation: int) -> List[float]:
        if self.mutation_method == "gaussian":
            return self.gaussian_mutation(individual, current_generation)
        elif self.mutation_method == "uniform":
            return self.uniform_mutation(individual)
        elif self.mutation_method == "boundary":
            return self.boundary_mutation(individual)
        return self.gaussian_mutation(individual, current_generation)

    def gaussian_mutation(self, individual: List[float], current_generation: int) -> List[float]:
        mutated = individual.copy()
        
        if not self.use_adaptive_mutation:
            mutation_rate = self.mutation_rate
            mutation_strength = 0.1
        else:
            mutation_rate = self.mutation_rate * (1 - current_generation / self.max_generations)
            mutation_strength = 0.5 * (1 - current_generation / self.max_generations)

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                delta = random.gauss(0, mutation_strength * (self.variable_bounds[i][1] - self.variable_bounds[i][0]))
                mutated[i] += delta
                mutated[i] = max(self.variable_bounds[i][0], min(mutated[i], self.variable_bounds[i][1]))

        return mutated

    def uniform_mutation(self, individual: List[float]) -> List[float]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.uniform(self.variable_bounds[i][0], self.variable_bounds[i][1])
        return mutated

    def boundary_mutation(self, individual: List[float]) -> List[float]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.choice([self.variable_bounds[i][0], self.variable_bounds[i][1]])
        return mutated

    def calculate_population_diversity(self, population: List[List[float]]) -> float:
        if not population:
            return 0.0
        mean_individual = np.mean(population, axis=0)
        diversity = np.mean([np.linalg.norm(np.array(ind) - mean_individual) for ind in population])
        return diversity

    def optimize(self, progress_bar=None) -> Tuple[List[float], float]:
        population = self.initialize_population()
        
        self.best_fitness_history = []
        self.best_solution_history = []
        self.mean_fitness_history = []
        self.diversity_history = []

        for generation in range(self.max_generations):
            if progress_bar:
                progress_bar.progress((generation + 1) / self.max_generations)
            
            fitness = self.calculate_fitness(population)
            diversity = self.calculate_population_diversity(population)
            best_fitness = max(fitness)
            mean_fitness = np.mean(fitness)
            best_solution = population[fitness.index(best_fitness)]
            
            self.best_fitness_history.append(best_fitness)
            self.best_solution_history.append(best_solution)
            self.mean_fitness_history.append(mean_fitness)
            self.diversity_history.append(diversity)
            
            num_elite = int(self.population_size * self.elitism_rate)
            elite_indices = sorted(range(len(fitness)), key=lambda k: fitness[k], reverse=True)[:num_elite]
            elite_population = [population[i] for i in elite_indices]
            
            parents = self.select_parents(population, fitness)
            
            next_population = elite_population.copy()
            while len(next_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.perform_mutation(c1, generation)
                c2 = self.perform_mutation(c2, generation)
                next_population.extend([c1, c2])
            
            population = next_population[:self.population_size]

        final_fitness = self.calculate_fitness(population)
        best_index = final_fitness.index(max(final_fitness))

        return population[best_index], max(final_fitness)

    def plot_performance(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        axs[0, 0].plot(self.best_fitness_history, label='Best Fitness')
        axs[0, 0].plot(self.mean_fitness_history, label='Mean Fitness')
        axs[0, 0].set_title('Fitness Progression')
        axs[0, 0].set_xlabel('Generation')
        axs[0, 0].set_ylabel('Fitness')
        axs[0, 0].legend()
        
        axs[0, 1].plot([sol[0] for sol in self.best_solution_history])
        axs[0, 1].set_title('Best Solution Progression')
        axs[0, 1].set_xlabel('Generation')
        axs[0, 1].set_ylabel('Best Solution Value')
        
        axs[1, 0].plot(self.diversity_history)
        axs[1, 0].set_title('Population Diversity')
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Diversity Metric')
        
        axs[1, 1].scatter(range(len(self.best_fitness_history)), self.best_fitness_history, alpha=0.7)
        axs[1, 1].set_title('Fitness Convergence')
        axs[1, 1].set_xlabel('Generation')
        axs[1, 1].set_ylabel('Best Fitness')
        
        plt.tight_layout()
        return fig

def predefined_functions() -> Dict[str, Callable]:
    return {
        "Sine Wave Modulation": lambda x: x * np.sin(10 * np.pi * x) + 1,
        "Quadratic": lambda x: -x**2,
        "Exponential": lambda x: np.exp(-x**2),
        "Absolute Cosine": lambda x: np.abs(np.cos(x))
    }

def app():
    st.title("🧬 Interactive Genetic Algorithm Optimizer")    
    st.sidebar.markdown("""
    ## Developed by **Pruthak Jani**
    [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat&logo=github)](https://github.com/pruthakjani5)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/pruthak-jani/)
    
    ## Welcome to the Interactive Genetic Algorithm Optimizer!
    
    ### What is a Genetic Algorithm? 🧬

    A Genetic Algorithm (GA) is a search heuristic inspired by Charles Darwin's theory of natural evolution. It mimics the process of natural selection to find optimal solutions to complex problems.

    #### Key Components:

    - **🧩 Population**: A set of potential solutions to the problem. Each individual in the population represents a possible solution.
    - **🔀 Selection**: The process of choosing the best-performing solutions (parents) based on their fitness scores to create offspring for the next generation.
    - **🔬 Crossover**: A genetic operator used to combine the genetic information of two parents to generate new offspring. Common methods include:
        - **Simulated Binary Crossover (SBX)**: Creates offspring by blending parent genes with a probability distribution.
        - **Single Point Crossover**: Splits parent genes at a single point and exchanges segments to create offspring.
        - **Uniform Crossover**: Randomly selects genes from either parent to create offspring.
    - **🎲 Mutation**: Introduces random variations to offspring genes to maintain genetic diversity. Common methods include:
        - **Gaussian Mutation**: Adds a random value from a Gaussian distribution to genes.
        - **Uniform Mutation**: Replaces genes with random values within the variable bounds.
        - **Boundary Mutation**: Sets genes to their minimum or maximum bounds.
    - **🏆 Elitism**: Preserves a fraction of the best-performing individuals from the current generation to the next generation to ensure the best solutions are not lost.
    - **📈 Adaptive Mutation**: Adjusts the mutation rate over generations to balance exploration and exploitation.

    #### Selection Methods:

    - **🏅 Tournament Selection**: Randomly selects a subset of individuals and chooses the best among them as parents.
    - **🎡 Roulette Wheel Selection**: Selects parents based on their fitness proportionate probability.
    - **📊 Rank Selection**: Ranks individuals by fitness and selects parents based on their rank.

    #### Why Use Genetic Algorithms? 🤔

    Genetic Algorithms are powerful tools for solving optimization problems where the search space is large, complex, or poorly understood. They are particularly useful when:
    - The problem has many local optima.
    - The objective function is noisy or discontinuous.
    - The search space is high-dimensional.

    By mimicking natural evolution, GAs can efficiently explore the search space and converge to high-quality solutions.
    
    ### About This App 🚀

    This app is designed to help you explore and understand the power of Genetic Algorithms (GAs) in solving complex optimization problems. Created with insights from the "Optimization Techniques" course, this tool provides an intuitive interface for experimenting with various GA parameters and observing their effects on optimization performance.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        function_choice = st.selectbox(
            "Select Optimization Function", 
            list(predefined_functions().keys()) + ["Custom Function"]
        )
        
        if function_choice == "Custom Function":
            custom_func = st.text_input(
                "Enter custom function (use x as variable)", 
                "x * np.sin(x)"
            )
            try:
                objective_function = eval(f"lambda x: {custom_func}")
                objective_function(1)
            except Exception as e:
                st.error(f"Invalid function: {e}")
                return
        else:
            objective_function = predefined_functions()[function_choice]

    with col2:
        min_range = st.number_input("Minimum Range", value=-5.0, step=0.1)
        max_range = st.number_input("Maximum Range", value=5.0, step=0.1)
        
        if min_range >= max_range:
            st.error("Minimum range must be less than maximum range")
            return

    # GA Configuration
    st.subheader("Algorithm Parameters")
    
    # Create three columns for different parameter groups
    col_basic, col_rates, col_methods = st.columns(3)
    
    # Basic parameters
    with col_basic:
        st.markdown("#### Basic Parameters")
        population_size = st.slider("Population Size", 50, 500, 200)
        max_generations = st.slider("Max Generations", 10, 300, 150)
        
    # Rate parameters
    with col_rates:
        st.markdown("#### Genetic Operators Rates")
        crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1)
        elitism_rate = st.slider("Elitism Rate", 0.01, 0.3, 0.1)
    
    # Method selection
    with col_methods:
        st.markdown("#### Genetic Operators")
        selection_method = st.selectbox(
            "Selection Method",
            ["tournament", "roulette", "rank"],
            help="Choose the method for selecting parents"
        )
        
        crossover_method = st.selectbox(
            "Crossover Method",
            ["simulated_binary", "single_point", "uniform"],
            help="Choose the method for combining parents"
        )
        
        mutation_method = st.selectbox(
            "Mutation Method",
            ["gaussian", "uniform", "boundary"],
            help="Choose the method for mutating offspring"
        )
        
    adaptive_mutation = st.checkbox("Use Adaptive Mutation", value=True,
                                  help="Automatically adjust mutation rate over generations")

    # Run Optimization
    if st.button("Optimize"):
        progress_bar = st.progress(0)
        
        # Create GA Instance with selected methods
        ga = GeneticAlgorithm(
            objective_function=objective_function,
            num_variables=1,
            variable_bounds=[(min_range, max_range)],
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_rate=elitism_rate,
            use_adaptive_mutation=adaptive_mutation,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method
        )

        # Run Optimization
        best_solution, best_fitness = ga.optimize(progress_bar)

        # Results Display
        st.subheader("🏆 Optimization Results")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Best Solution", f"{best_solution[0]:.4f}")
        with col_res2:
            st.metric("Best Fitness", f"{best_fitness:.4f}")

        # Performance Visualization
        st.subheader("Performance Metrics")
        fig_performance = ga.plot_performance()
        st.pyplot(fig_performance)

        # Objective Function Visualization
        st.subheader("Objective Function")
        fig_obj = plt.figure(figsize=(10, 5))
        x = np.linspace(min_range, max_range, 200)
        y = [objective_function(xi) for xi in x]
        
        plt.plot(x, y, label='Objective Function')
        plt.scatter(best_solution[0], best_fitness, color='red', marker='*', s=200, label='GA Solution')
        plt.title('Objective Function Visualization')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        
        st.pyplot(fig_obj)

if __name__ == "__main__":
    st.set_page_config(page_title="Genetic Algorithm Optimizer", page_icon="🧬", layout="wide")
    app()
