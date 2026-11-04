import numpy as np
import pandas as pd
# Generate 10,000 random values from normal distribution
population = np.random.normal(loc=50, scale=10, size=10000)

# Population statistics
population_mean = np.mean(population)
population_std = np.std(population)

# Randomly select 30 samples
sample = np.random.choice(population, size=30, replace=False)

# Sample statistics
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)

# Standard error
standard_error = sample_std / np.sqrt(30)


df = pd.DataFrame({
    "Statistics": ["Mean", "Standard Deviation", "Standard Error"],
    "Population": [population_mean, population_std, "-"],
    "Sample": [sample_mean, sample_std, standard_error]
})
print(df)