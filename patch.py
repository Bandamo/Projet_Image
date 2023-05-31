import cv2
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

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

    def compute_dat_term(self, mask, method='max_gradient', only_isophote=False, plot=False, verbose=True):
        closest_pixel = self.get_closest_pixel(mask, self.position)
        
        grad = self.compute_gradient(mask)

        if method == 'closest_pixel':
            #print("Closest pixel: ", closest_pixel)
            isophote = np.array(grad[:1][closest_pixel[0], closest_pixel[1]])
        elif method == 'max_gradient':
            max_coord = np.unravel_index(np.argmax(np.sqrt(grad[0]**2 + grad[1]**2)), grad[0].shape)
            isophote = np.array([grad[0][max_coord], grad[1][max_coord]])
        
        isophote_T = self.perpendicular_vector(isophote)
        # We then need to get the normal vector to the contour at position
        normal = self.compute_normal(mask, closest_pixel)

        # We then compute the dot product between the two vectors

        if plot:
            plt.imshow(self.data)
            plt.plot(max_coord[1], max_coord[0], 'b*')
            plt.quiver(max_coord[1], max_coord[0], isophote_T[1], isophote_T[0], color='green')
            plt.quiver(max_coord[1], max_coord[0], normal[1], normal[0], color='yellow')
            plt.show()
        
        if only_isophote:
            self.dat_term = np.linalg.norm(abs(isophote_T)/255)
        else:
            self.dat_term = np.linalg.norm(abs(np.dot(isophote_T, normal)/255))

        if verbose:
            print("Closest pixel: ", closest_pixel)
            print("Isophote: ", isophote_T)
            print("Normal: ", normal)
            print('Dataterm: ', self.dat_term)
        
        return self.dat_term

    def compute_normal(self, mask, position):
        # Compute the normal vector to the contour at position
        mask = mask[self.position[0] - self.radius:self.position[0] + self.radius, self.position[1] - self.radius:self.position[1] + self.radius]
        normal = np.gradient(mask)

        position[0] = position[0] - self.position[0] + self.radius
        position[1] = position[1] - self.position[1] + self.radius

        # Evaluate the gradient at position
        normal = np.array([-normal[1][position[0], position[1]], normal[0][position[0], position[1]]])
        print("Normal: ", normal)
        return normal
    
    def compute_gradient(self, mask, plot=False):
        # Compute the gradient of the patch at position
        data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        grad = np.gradient(data)

        # Set the gradient to 0 if the pixel is in the mask
        mask = mask[self.position[0] - self.radius:self.position[0] + self.radius + 1, self.position[1] - self.radius:self.position[1] + self.radius + 1]
        grad[0][mask == 0] = 0
        grad[1][mask == 0] = 0
        
        # Smooth the gradient
        grad[0] = cv2.GaussianBlur(grad[0], (7,7), 0)
        grad[1] = cv2.GaussianBlur(grad[1], (7,7), 0)


        if plot:
            plt.figure()
            plt.imshow(data)

            plt.figure()
            plt.quiver(-grad[1], grad[0])
            plt.gca().invert_yaxis()
            plt.show()
        return grad

    def get_closest_pixel(self, mask, position):
        # Returns the closest pixel from the position to the contour
        min_dist = None
        closest_pixel = None
        for i in range(position[0] - self.radius, position[0] + self.radius):
            for j in range(position[1] - self.radius, position[1] + self.radius):
                if mask[i,j] != mask[position[0], position[1]]:
                    dist = np.linalg.norm(np.array([i,j]) - position)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        closest_pixel = [i,j]
        return closest_pixel

    def update_priority(self, mask, method='closest_pixel'):
        self.conf = self.compute_conf(mask)
        #print('Conf: %f' % self.conf)
        self.dat_term = self.compute_dat_term(mask, method)
        #print('Dat term: %s' % self.dat_term)
        self.priority = self.dat_term*self.conf
        return self.priority