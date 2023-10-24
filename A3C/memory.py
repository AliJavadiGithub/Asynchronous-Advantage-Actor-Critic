class Memory:
    def __init__(self):
        self.values = []
        self.log_probs = []
        self.rewards = []

    def clear_memory(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
    
    def remember(self, reward, value, log_prob):
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def sample_memory(self):
        return self.rewards, self.values, self.log_probs
    