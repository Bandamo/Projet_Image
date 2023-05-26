import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy as sp
from patch import Patch


class Main():
    def __init__(self) -> None:
        # Paramètres de l'image
        self.image = None
        self.arr = None
        self.shape = None

        self.contour = []
        self.mask = None

        self.patches = []
    
    # ------------------------------------ CONTOUR ------------------------------------
    def find_contour(self, plot = False, smoothing = False):
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

        if smoothing:
            self.contour = self.smoothing_contour(self.contour)

        if plot:
            # Plot
            plt.imshow(self.arr)
            plt.plot([c[1] for c in contour], [c[0] for c in contour], 'r.')
            plt.show()

    def smoothing_contour(self, contour):
        index = 0
        while index<len(contour):
            # Find neighbour of this point
            neighbour = []
            possible_neighbour = []
            for i in range(-1,2):
                for j in range(-1,2):
                    if not(i==0 and j==0):
                        possible_neighbour.append((contour[index][0]+i, contour[index][1]+j))

            for e in contour:
                if e in possible_neighbour:
                    neighbour.append(e)
            
            # Only keep 2 neighbour in contour
            if len(neighbour) > 2:
                i = np.random.randint(0, len(neighbour))
                neighbour.remove(neighbour[i])
                i = np.random.randint(0, len(neighbour))
                neighbour.remove(neighbour[i])
                for e in neighbour:
                    contour.remove(e)
            else:
                index += 1
        return contour           

    def load_mask(self,path):
        img = Image.open(path)
        m = np.asarray(img)
        m = m[:,:,0]
        self.mask = np.zeros(self.shape)
        self.mask[m > 0] = 1
        self.mask = self.mask[:,:,0]

    def update_contour(self, patch):
        # patch = (center, radius)
        center, radius = (patch.position, patch.radius)

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
                #print(str(self.contour[k]) + " removed")
                pass
        
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

    # ------------------------------------ IMAGE ------------------------------------

    def load_image(self, path):
        self.image =Image.open(path)
        self.arr = np.asarray(self.image)
        self.shape = self.arr.shape

        self.mask = np.ones(self.shape)
    
    def print_image(self):
        plt.imshow(self.arr)
        plt.show()
    
    def create_patches(self, patch_size):
        # Return a list of patches
        patches = []

        radius = int((patch_size-1)/2)
        hsize  = int(self.arr.shape[0]/patch_size)
        vsize  = int(self.arr.shape[1]/patch_size)

        # Get the datas
        for i in range(hsize):
            for j in range(vsize):
                data = self.arr[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                position = (i*patch_size+radius, j*patch_size+radius)
                patches.append(Patch(data=data, position=position, radius=radius))
        
        self.patches = patches
    
    # ------------------------------------ PROPAGATING TEXTURE ------------------------
    def find_best_patch(self, patch):
        # Return the best patch to replace the given one
        # patch : Patch
        # Return : Patch
        best_patch = None
        best_distance = float("inf")
        for p in self.patches:
            if p.conf >=1: # Condition on confidence
                distance = np.sum(np.square(p.data - patch.data))
                if distance < best_distance:
                    best_distance = distance
                    best_patch = p
        return best_patch

    def propagate_texture(self):
        list_id_priority = np.array([[k, self.patches[k].priority] for k in range(len(self.patches))])
        list_id_priority.sort(axis=1)

        for index in range(len(self.patches)):
            i = list_id_priority[index][0]
            if self.patches[i].active:
                # Find best patch
                best_patch = self.find_best_patch(self.patches[i])

                # Replace patch
                self.patches[i].data = best_patch.data
                self.patches[i].active = False

                # Update mask
                self.update_contour(self.patches[i])

                # Update image
                pos = self.patches[i].position
                radius = self.patches[i].radius
                hindex = (pos[0]-radius, pos[0]+radius)
                vindex = (pos[1]-radius, pos[1]+radius)
                self.arr[hindex[0]:hindex[1], vindex[0]:vindex[1]] = self.patches[i].data

    
if __name__=="__masqfin__":
    m = Main()
    m.load_image("image.jpg")
    m.load_mask("mask.ppm")
    m.find_contour(plot = False, smoothing = True)
    m.update_contour(((143,239), 10))

if __name__=="__main__":
    a = Image.open("image.jpg")
    a = np.asarray(a)
    
    m = Main()
    m.arr = a
    m.create_patches(50)