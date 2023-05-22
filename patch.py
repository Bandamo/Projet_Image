import numpy as np
import scipy as sc

class Patch():
    def __init__(self, data, radius, position, initial_conf=0, initial_dat_term=1) -> None:
        self.data = data
        self.radius = radius
        self.position = position
        self.conf = initial_conf
        self.dat_term = initial_dat_term
        self.active = False 
        pass

    def perpendicular_vector(vector):
        # Returns a perpendicular vector to the input vector
        return np.array([-vector[1], vector[0]])

    def compute_conf(self, contour, mask):
        # Compute the confidence term
        for i in range(self.position[0] - self.radius, self.position[0] + self.radius):
            for j in range(self.position[1] - self.radius, self.position[1] + self.radius):
                if mask[i,j] == 0:
                    self.conf += 1
        self.conf /= (2*self.radius + 1)**2
        return self.conf

    def compute_dat_term(self, data, position, contour):
        # We first need to compute the isophote at position
        pass
    
    def compute_gradient(self, data, position):
        # Compute the gradient of the patch at position
        grad = np.gradient(data)[position[0], position[1], []]
    def update_priority(self, contour):
        self.priority = self.conf*self.dat_term
        return self.priority