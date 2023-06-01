import cv2
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2lab

class Patch():
    """
    Class representing a patch in the image

    Attributes:
        data (np.array): The image data of the patch
        radius (int): The radius of the patch
        position (tuple): The position of the patch in the image
        conf (float): The confidence term of the patch
        dat_term (float): The data term of the patch
        active (bool): Whether the patch is active or not (is in the contour)
        priority (float): The priority of the patch

    Methods:
        set_state(state): Sets the state of the patch (active or not)
        is_active(): Returns whether the patch is active or not
        perpendicular_vector(vector): Returns a perpendicular vector to the input vector
        compute_conf(mask): Computes the confidence term of the patch
        compute_dat_term(mask, method, only_isophote, plot, verbose): Computes the data term of the patch
        compute_normal(mask, position): Computes the normal vector to the contour at position
        compute_gradient(mask, plot): Computes the gradient of the patch
        get_closest_pixel(mask, position): Returns the closest pixel from the position to the contour
        update_priority(mask, method): Updates the priority of the patch


    """
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
        """
        Sets the state of the patch (active or not)
        """
        self.active = state

    def is_active(self):
        """
        Returns whether the patch is active or not
        """
        return self.active

    def perpendicular_vector(self, vector):
        """
        Returns a perpendicular vector to the input vector
        """
        return np.array([-vector[1], vector[0]])

    def compute_conf(self, mask):
        """
        Compute the confidence term of the patch
        """
        for i in range(self.position[0] - self.radius, self.position[0] + self.radius):
            for j in range(self.position[1] - self.radius, self.position[1] + self.radius):
                if mask[i,j] == 1:
                    self.conf += 1
        self.conf /= (2*self.radius + 1)**2
        return self.conf

    def compute_dat_term(self, mask, method='max_gradient', only_isophote=False, plot=False, verbose=False):
        """
        Compute the data term of the patch

        Parameters:
            mask (np.array): The mask of the image
            method (str): The method to use to compute the data term
                - 'closest_pixel': Take the value of the gradient at the closest pixel to the contour
                - 'max_gradient': Take the value of the gradient at the pixel with the highest gradient
            only_isophote (bool): Whether to only use the isophote or use the combo with the normal vector to the contour
            plot (bool): Whether to plot gradient, isophote^T and normal vector
            verbose (bool): Whether to print the closest pixel, isophote, normal vector and data term
        """
        closest_pixel = self.get_closest_pixel(mask, self.position)
        closest_pixel_org = closest_pixel.copy()
        closest_pixel_org[0] = closest_pixel[0] - self.position[0] + self.radius
        closest_pixel_org[1] = closest_pixel[1] - self.position[1] + self.radius

        grad = self.compute_gradient(mask, plot)

        if method == 'closest_pixel':
            #print("Closest pixel: ", closest_pixel)
            isophote = np.array([grad[0][closest_pixel_org[0], closest_pixel_org[1]], grad[1][closest_pixel_org[0], closest_pixel_org[1]]])
        elif method == 'max_gradient':
            max_coord = np.unravel_index(np.argmax(np.sqrt(grad[0]**2 + grad[1]**2)), grad[0].shape)
            isophote = np.array([grad[0][max_coord], grad[1][max_coord]])
        elif method == 'mean_gradient':
            isophote = np.array([np.mean(grad[0]), np.mean(grad[1])])
        
        isophote_T = self.perpendicular_vector(isophote)
        # We then need to get the normal vector to the contour at position
        normal = self.compute_normal(mask, closest_pixel)

        if plot:
            if method == 'closest_pixel' or method == 'mean_gradient':
                plot_coord = closest_pixel
            elif method == 'max_gradient':
                plot_coord = max_coord

            plt.plot(plot_coord[1], plot_coord[0], 'b*')
            plt.quiver(plot_coord[1], plot_coord[0], -isophote_T[1], isophote_T[0], color='red')
            plt.quiver(plot_coord[1], plot_coord[0], -normal[1], normal[0], color='blue')
            plt.show()
        
        if only_isophote:
            self.dat_term = np.linalg.norm(abs(isophote_T))
        else:
            self.dat_term = np.linalg.norm(abs(np.dot(isophote_T, normal)))

        if verbose:
            print("Closest pixel: ", closest_pixel)
            print("Isophote: ", isophote_T)
            print("Normal: ", normal)
            print('Dataterm: ', self.dat_term)
        
        return self.dat_term

    def compute_normal(self, mask, position):
        """
        Compute the normal vector to the contour at position
        """
        mask = mask[self.position[0] - self.radius:self.position[0] + self.radius + 1, self.position[1] - self.radius:self.position[1] + self.radius + 1]
        normal = np.gradient(mask)

        position[0] = position[0] - self.position[0] + self.radius
        position[1] = position[1] - self.position[1] + self.radius

        # Evaluate the gradient at position
        normal = np.array([normal[0][position[0], position[1]], normal[1][position[0], position[1]]])
        return normal
    
    def compute_gradient(self, mask, plot=False):
        """
        Compute the gradient of the patch
        """
        data = rgb2gray(self.data)

        mask = mask[self.position[0] - self.radius:self.position[0] + self.radius + 1, self.position[1] - self.radius:self.position[1] + self.radius + 1]

        data[mask == 0] = None

        gradient = np.nan_to_num(np.array(np.gradient(data)))

        # Smooth the gradient
        gradient[0] = cv2.GaussianBlur(gradient[0], (3,3), 0)
        gradient[1] = cv2.GaussianBlur(gradient[1], (3,3), 0)

        if plot:
            plt.figure()
            plt.imshow(self.data)
            plt.figure()
            plt.imshow(data)

            plt.quiver(-gradient[1], gradient[0])
        return gradient

    def get_closest_pixel(self, mask, position):
        """
        Returns the closest pixel from the position to the contour
        """
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

    def update_priority(self, mask, method='max_gradient', only_isophote=False, plot=False, verbose=False):
        """
        Updates the priority of the patch
        """
        self.conf = self.compute_conf(mask)
        #print('Conf: %f' % self.conf)
        self.dat_term = self.compute_dat_term(mask, method, only_isophote, plot, verbose)
        #print('Dat term: %s' % self.dat_term)
        self.priority = self.conf * self.dat_term

        return self.priority