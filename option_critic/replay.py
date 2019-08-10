import numpy as np
import random


class Replay:
    def __init__(self, capacity, batch_size, to_numpy=True):
        self.buffer = []
        self.capacity = int(capacity)
        self.to_numpy = to_numpy
        self.batch_size = int(batch_size)
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def add_batch(self, experiences):
        for experience in experiences:
            self.add(experience)

    def add_vec(self, experiences):
        experiences = zip(*experiences)
        self.add_batch(experiences)

    def sample(self, batch_size=None):
        if not len(self.buffer):
            return None, None

        if batch_size is None:
            batch_size = self.batch_size

        sampled_data = random.sample(self.buffer, batch_size)
        sampled_data = zip(*sampled_data)
        if self.to_numpy:
            sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def __len__(self):
        return len(self.buffer)


