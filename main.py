import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Main():
    def __init__(self) -> None:
        # Paramètres de l'image
        self.image = None
        self.arr = None
        self.shape = None

        self.contour = []
        self.mask = None
    
    def find_contour(self):
        # Trouve un contour grossier
        pass

    def load_image(self, path):
        self.image =Image.open(path)
        self.arr = np.asarray(self.image)
        self.shape = self.arr.shape

        self.mask = np.ones(self.shape)
    
    def debug_mask(self):
        pass

    def print_image(self):
        plt.imshow(self.arr)
        plt.show()
    
    
if __name__=="__main__":
    m = Main()
    #m.load_image("image.jpg")
    m.mask = np.ones((430,287))
    m.mask[211:296,157:233,:,:,:] = np.zeros((3,1))
    plt.imshow(m.mask)
    plt.show()
    
