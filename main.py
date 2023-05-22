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
        self.contour = contour

        if plot:
            # Plot
            plt.imshow(self.arr)
            plt.plot([c[1] for c in contour], [c[0] for c in contour], 'r.')
            plt.show()

    def load_mask(self,path):
        img = Image.open(path)
        m = np.asarray(img)
        m = m[:,:,0]
        self.mask = np.zeros(self.shape)
        self.mask[m > 0] = 1
        self.mask = self.mask[:,:,0]

    def update_contour(self, patch):
        # patch = (center, radius)
        center, radius = patch

        # All point in a square of size 2*radius
        l_point_in_patch = []
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                l_point_in_patch.append((center[0]+i, center[1]+j))
        
        # Remove point in the patch from the contour
        contour = []
        for k in range(len(self.contour)):
            if not(self.contour[k] in l_point_in_patch):
                contour.append(self.contour[k])
            else:
                print(str(self.contour[k]) + " removed")
        
        # Add patch border to the contour
        patch_border = []
        for i in range(-radius, radius):
            if self.mask[center[0]+i, center[1]+radius] == 0:
                patch_border.append((center[0]+i, center[1]+radius))
            if self.mask[center[0]+i, center[1]-radius] == 0:
                patch_border.append((center[0]+i, center[1]-radius))
        for j in range(-radius, radius):
            if self.mask[center[0]+radius, center[1]+j] == 0:
                patch_border.append((center[0]+radius, center[1]+j))
            if self.mask[center[0]-radius, center[1]+j] == 0:
                patch_border.append((center[0]-radius, center[1]+j))

        contour = contour + patch_border
        self.contour = contour

        # Update mask
        only_patch = np.zeros(self.shape)
        only_patch = only_patch[:,:,0]
        only_patch[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius] = 1
        self.mask = np.logical_or(self.mask, only_patch)
    
    def load_image(self, path):
        self.image =Image.open(path)
        self.arr = np.asarray(self.image)
        self.shape = self.arr.shape

        self.mask = np.ones(self.shape)
    
    def print_image(self):
        plt.imshow(self.arr)
        plt.show()
    
    
if __name__=="__main__":
    m = Main()
    m.load_image("image.jpg")
    m.load_mask("mask.ppm")
    m.find_contour(True)
    plt.imshow(m.mask)
    plt.show()
    m.update_contour(((143,239), 10))
    plt.imshow(m.mask)
    plt.show()
    plt.imshow(m.arr)
    plt.plot([c[1] for c in m.contour], [c[0] for c in m.contour], 'r.')
    plt.show()
