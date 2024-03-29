import math

conditional_probabilities = {
    60: {0: 0.06, 63: 0.01},
    30: {0: 0.05, 63: 0.0002},
    0: {0: 0.08, 63: 0.0003},
    -30: {0: 0.07, 63: 0.0004},
    -60: {0: 0.04, 63: 0.0005}
}

# Function to compute log likelihood for a given angle and y-locations of dots
def compute_log_likelihood(angle, conditional_probabilities, dot_distribution):
    log_likelihood = 0.0
    for y, prob in dot_distribution.items():
        if y in conditional_probabilities[angle]:
            log_likelihood += math.log(conditional_probabilities[angle][y] * prob)
    return log_likelihood

# Function to compute log posterior probabilities
def compute_log_posterior(conditional_probabilities, dot_distribution):
    log_posteriors = {}
    total_log_likelihood = float('-inf') # Initialize with negative infinity
    for angle in conditional_probabilities:
        log_likelihood = compute_log_likelihood(angle, conditional_probabilities, dot_distribution)
        log_posteriors[angle] = log_likelihood
        total_log_likelihood = math.log(math.exp(total_log_likelihood) + math.exp(log_likelihood))
    # Normalize log likelihoods
    for angle in log_posteriors:
        log_posteriors[angle] -= total_log_likelihood
    return log_posteriors

# Example image dot distribution (replace with your actual distribution)
image_dot_distribution = {0: 0.1, 63: 0.9}  # Example: equal probability for y=0 and y=63

# Compute log posterior probabilities
log_posterior_probabilities = compute_log_posterior(conditional_probabilities, image_dot_distribution)

# Print log posterior probabilities
print("Log Posterior Probabilities:")
for angle, log_probability in log_posterior_probabilities.items():
    print(f"Angle: {angle}, Log Probability: {log_probability}")
