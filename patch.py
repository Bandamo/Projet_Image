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
        self.priority = 0
        pass
    
    def set_state(self, state: bool):
        self.active = state

    def is_active(self):
        return self.active

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
        closest_pixel = self.get_closest_pixel(contour, position)

        # We first need to compute the isophote at position
        isophote_T = self.perpendicular_vector(self.compute_gradient(data, closest_pixel))
        # We then need to get the normal vector to the contour at position
        normal = self.compute_normal(contour, closest_pixel)
        # We then compute the dot product between the two vectors
        self.dat_term = np.dot(isophote_T, normal)
        return self.dat_term

    def compute_normal(self, contour, position):
        # Compute the normal vector to the contour at position
        normal = np.gradient(contour)[:1, position[0], position[1]]
        return normal
    
    def compute_gradient(self, data, position):
        # Compute the gradient of the patch at position
        grad = np.gradient(data)[:1, position[0], position[1]]
        return grad

    def get_closest_pixel(self, contour, position):
        # Returns the closest pixel from the position to the contour
        min_dist = None
        for i in range(position[0] - self.radius, position[0] + self.radius):
            for j in range(position[1] - self.radius, position[1] + self.radius):
                if contour[i,j] == 1:
                    dist = np.linalg.norm(np.array([i,j]) - position)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
        
        return min_dist

    def update_priority(self, contour):
        self.conf = self.compute_conf(contour, self.position)
        self.dat_term = self.compute_dat_term(self.data, self.position, contour)
        self.priority = self.conf*self.dat_term
        return self.priority