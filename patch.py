import cv2
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

    def perpendicular_vector(self, vector):
        # Returns a perpendicular vector to the input vector
        return np.array([-vector[1], vector[0]])

    def compute_conf(self, mask):
        # Compute the confidence term
        for i in range(self.position[0] - self.radius, self.position[0] + self.radius):
            for j in range(self.position[1] - self.radius, self.position[1] + self.radius):
                if mask[i,j] == 1:
                    self.conf += 1
        self.conf /= (2*self.radius + 1)**2
        return self.conf

    def compute_dat_term(self, mask, method='closest_pixel'):
        closest_pixel = self.get_closest_pixel(mask, self.position)
        
        grad = self.compute_gradient()

        if method == 'closest_pixel':
            isophote = np.array([grad[closest_pixel[0], closest_pixel[1], 0], grad[closest_pixel[0], closest_pixel[1], 1]])
        elif method == 'max_gradient':
            max_coord = np.unravel_index(np.argmax(np.sqrt(grad[0]**2 + grad[1]**2)), grad[0].shape)
            isophote = np.array([grad[0][max_coord], grad[1][max_coord]])
        
        isophote_T = self.perpendicular_vector(isophote)
        # We then need to get the normal vector to the contour at position
        normal = self.compute_normal(mask, closest_pixel)
        # We then compute the dot product between the two vectors
        self.dat_term = abs(np.dot(isophote_T, normal)/255)

        return self.dat_term

    def compute_normal(self, mask, position):
        # Compute the normal vector to the contour at position
        normal = np.gradient(mask)[:1, position[0], position[1]]
        return normal
    
    def compute_gradient(self):
        # Compute the gradient of the patch at position
        data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        data = cv2.GaussianBlur(data, (7,7), 0)

        grad = np.gradient(data)
        return grad

    def get_closest_pixel(self, mask, position):
        # Returns the closest pixel from the position to the contour
        min_dist = None
        closest_pixel = None
        for i in range(position[0] - self.radius, position[0] + self.radius):
            for j in range(position[1] - self.radius, position[1] + self.radius):
                if mask[i,j] != mask[position]:
                    dist = np.linalg.norm(np.array([i,j]) - position)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
        
        print("Min dist: %f"%min_dist)
        print("Closest pixel: %s"%closest_pixel)
        return closest_pixel

    def update_priority(self, mask, method='closest_pixel'):
        self.conf = self.compute_conf(mask, self.position)
        
        self.dat_term = self.compute_dat_term(mask, method)
        self.priority = self.conf*self.dat_term
        return self.priority