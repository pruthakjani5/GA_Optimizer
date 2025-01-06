# 🧬 Genetic Algorithm Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genetic-algo-optimizer.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-1.21+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-blue.svg)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/pruthak-jani/)

An interactive web application that demonstrates the power of genetic algorithms through visual optimization. This tool allows users to experiment with different genetic algorithm parameters and observe their effects on optimization performance in real-time.

## 🌟 Features

- **Interactive Parameter Tuning**: Customize all aspects of the genetic algorithm:
  - Population size and maximum generations
  - Crossover, mutation, and elitism rates
  - Selection, crossover, and mutation methods
  - Adaptive mutation capabilities

- **Multiple Optimization Functions**:
  - Sine Wave Modulation
  - Quadratic Function
  - Exponential Function
  - Absolute Cosine
  - Custom Function Support

- **Real-time Visualization**:
  - Fitness progression over generations
  - Population diversity metrics
  - Best solution convergence
  - Objective function landscape

- **Advanced Genetic Operators**:
  - Selection Methods: Tournament, Roulette Wheel, Rank
  - Crossover Methods: Simulated Binary, Single Point, Uniform
  - Mutation Methods: Gaussian, Uniform, Boundary

## 🚀 Live Demo

Try out the live application: [Genetic Algorithm Optimizer](https://genetic-algo-optimizer.streamlit.app/)

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/pruthakjani5/GA_Optimizer.git

# Navigate to the project directory
cd GA_Optimizer

# Install required dependencies
pip install -r requirements.txt

# Run the application
streamlit run ga.py
```

## 📖 How It Works

The Genetic Algorithm Optimizer implements the following key components:

1. **Initialization**: Creates an initial population of random solutions within specified bounds

2. **Evolution Process**:
   - **Selection**: Identifies promising solutions using various selection methods
   - **Crossover**: Combines parent solutions to create offspring
   - **Mutation**: Introduces random variations to maintain diversity
   - **Elitism**: Preserves the best solutions across generations

3. **Visualization**: Tracks and displays:
   - Fitness improvements
   - Solution diversity
   - Convergence patterns
   - Optimization landscape

## 🎮 Usage Example

```python
# Initialize the genetic algorithm
ga = GeneticAlgorithm(
    objective_function=lambda x: x * np.sin(10 * np.pi * x) + 1,
    num_variables=1,
    variable_bounds=[(-5, 5)],
    population_size=200,
    max_generations=150
)

# Run optimization
best_solution, best_fitness = ga.optimize()

# Visualize results
ga.plot_performance()
```

## 🔧 Parameters Explained

- **Population Size**: Number of solutions in each generation
- **Max Generations**: Total iterations of the evolutionary process
- **Crossover Rate**: Probability of combining parent solutions (0.0-1.0)
- **Mutation Rate**: Probability of random solution modification (0.0-1.0)
- **Elitism Rate**: Proportion of best solutions preserved (0.0-1.0)

## 📈 Advanced Features

### Selection Methods
- **Tournament**: Randomly selects groups and picks the best from each
- **Roulette**: Selection probability proportional to fitness
- **Rank**: Selection based on fitness ranking

### Crossover Methods
- **Simulated Binary**: Creates offspring using a probability distribution
- **Single Point**: Exchanges genetic material at a random point
- **Uniform**: Randomly selects genes from either parent

### Mutation Methods
- **Gaussian**: Adds random values from a normal distribution
- **Uniform**: Replaces genes with random values
- **Boundary**: Sets genes to their bounds

## 📚 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## ⭐ Show your support

Give a ⭐️ if this project helped you!
