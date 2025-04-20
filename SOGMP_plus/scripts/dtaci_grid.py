import numpy as np

vec_zero_min = np.vectorize(lambda x: min(x, 0))

def pinball(u, alpha):
    return alpha * u - vec_zero_min(u)

class DtACI:
    def __init__(self, alpha=0.1, gammas=np.array([0.05, 0.1, 0.2]), 
                 sigma=1/1000, eta=2.72, initial_pred=0.5):
        self.alpha = alpha
        self.gammas = gammas
        self.sigma = sigma
        self.eta = eta
        self.true_values = []
        self.predictions = []
        self.errors = []
        self.gamma_sequence = []
        self.weighted_average_prediction_sequence = []
        self.coverage = []
        self.num_experts = len(gammas)
        self.initial_pred = initial_pred
        self.expert_predictions = np.full(self.num_experts, initial_pred)
        self.expert_weights = np.ones(self.num_experts)
        self.current_expert = np.random.choice(np.arange(self.num_experts))        
        self.expert_cumulative_losses = np.zeros(self.num_experts)
        self.expert_probs = np.full(self.num_experts, 1 / self.num_experts)
        self.last_prediction = initial_pred

    def make_prediction(self):
        if len(self.true_values) == 0:
            self.last_prediction = self.initial_pred
            return self.initial_pred
        else:
            prediction = self.expert_predictions[self.current_expert]
            self.last_prediction = prediction
            return prediction

    def update_true_value(self, new_true_value):
        self.true_values.append(new_true_value)
        truth = new_true_value
        prediction = self.last_prediction

        self.predictions.append(prediction)
        self.gamma_sequence.append(self.gammas[self.current_expert])
        self.weighted_average_prediction_sequence.append(np.sum(self.expert_probs * self.expert_predictions))

        expert_losses = pinball(truth - self.expert_predictions, self.alpha)
        self.expert_predictions -= self.gammas * (self.alpha - (self.expert_predictions < truth).astype(float))
        
        if self.eta < float('inf'):
            expert_bar_weights = self.expert_weights * np.exp(-self.eta * expert_losses)
            expert_next_weights = (1 - self.sigma) * expert_bar_weights / np.sum(expert_bar_weights) + self.sigma / self.num_experts
            self.expert_probs = expert_next_weights / np.sum(expert_next_weights)
            self.current_expert = np.random.choice(np.arange(self.num_experts), p=self.expert_probs)
            self.expert_weights = expert_next_weights
        else:
            self.expert_cumulative_losses += expert_losses
            self.current_expert = np.argmin(self.expert_cumulative_losses)
            
        self.errors.append(prediction - truth)
        self.coverage.append(float(prediction >= truth))

class DtACIGrid:
    def __init__(self, grid_height, grid_width, alpha=0.1, gammas=np.array([0.05, 0.1, 0.2]), sigma=1/1000, eta=2.72, initial_pred=0.5):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.alpha = alpha
        self.gammas = gammas
        self.sigma = sigma
        self.eta = eta
        self.initial_pred = initial_pred
        
        # Initialize a DtACI estimator for each cell in the grid
        self.dtaci_grid = np.array([[DtACI(alpha, gammas, sigma, eta, initial_pred) 
                                     for _ in range(grid_width)] 
                                     for _ in range(grid_height)])

    def make_prediction(self):
        # Generate predictions for each grid cell
        pred_grid = np.zeros((self.grid_height, self.grid_width))
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                pred_grid[i, j] = self.dtaci_grid[i, j].make_prediction()
        return pred_grid

    def update_true_grid(self, true_grid):
        # Update each grid cell with its true occupancy value
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                self.dtaci_grid[i, j].update_true_value(true_grid[i, j])

    def get_prediction_grid(self):
        pred_grid = np.zeros((self.grid_height, self.grid_width))
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                pred_grid[i, j] = self.dtaci_grid[i, j].last_prediction
        return pred_grid
    
    def get_error_grid(self):
        error_grid = np.zeros((self.grid_height, self.grid_width))
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                error_grid[i, j] = self.dtaci_grid[i, j].errors[-1]
        return error_grid
    
    def get_coverage_grid(self):
        coverage_grid = np.zeros((self.grid_height, self.grid_width))
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                coverage_grid[i, j] = self.dtaci_grid[i, j].coverage[-1]
        return coverage_grid
