import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy as sp


class Main():
    def __init__(self) -> None:
        # Paramètres de l'image
        self.image = None
        self.arr = None
        self.shape = None

        self.contour = []
        self.mask = None
    
    def find_contour(self, plot = False):
        # To binary 
        m = np.zeros(self.shape)
        m[self.mask > 0] = 1

        # Gradient
        g = np.gradient(m)
        gp = abs(g[0]) + abs(g[1])
        gp = gp[:,:,0]
        gp = np.minimum(2*gp, 1)

        # Contour
        list_contour = np.where(gp[:,:] == 1)
        contour = [(list_contour[0][k], list_contour[1][k]) for k in range(len(list_contour[0]))]

        if plot:
            # Plot
            plt.imshow(self.arr)
            plt.plot([c[1] for c in contour], [c[0] for c in contour], 'r.')
            plt.show()

    def load_image(self, path):
        self.image =Image.open(path)
        self.arr = np.asarray(self.image)
        self.shape = self.arr.shape

        self.mask = np.ones(self.shape)
    
    def load_mask(self,path):
        img = Image.open(path)
        self.mask = np.asarray(img)
        
    def print_image(self):
        plt.imshow(self.arr)
        plt.show()
    
    
if __name__=="__main__":
    m = Main()
    m.load_image("image.jpg")
    m.load_mask("mask.ppm")
    m.find_contour()

