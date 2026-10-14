# MATLAB scripts (thesis chapter 4)

These are the original thesis scripts for the GA-BP neural-network model, kept as they were written, except that some files have been renamed. They read `四向量.xlsx`, which has the same content as `data/scoring_memberships.csv`. The file paths inside the scripts are from my original machine and need to be updated before running.

| File | Purpose |
|---|---|
| `hidden_layer_selection.m` | Compares 2–15 hidden neurons on MSE, MAE and accuracy |
| `find_optimal_hidden_neurons_multirun.m` | Repeats the hidden-layer comparison over multiple runs |
| `bp_baseline.m` | BP network (4–9–1) without GA initialisation |
| `gabp.m` | Genetic algorithm (population 30, 30 generations, crossover 0.7, mutation 0.1) sets the initial weights, followed by BP training |
| `evaluate_fitness.m` | Fitness function used by the GA (MSE-based) |

**Note:** the model's target level is, by definition, the largest of its four inputs, so the network learns a known rule rather than predicting anything new. See Appendix B of the portfolio and `analysis/05_robustness_checks.py`.
