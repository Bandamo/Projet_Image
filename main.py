import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy as sp
from patch import Patch
import tqdm
import time
import os

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
        #create 2D array of 0 and 1
        self.mask = self.mask[:,:,0]
        print("Shape of mask : " + str(self.mask.shape))

    def update_contour(self, patch, plot = False):
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
        for i in range(-radius-1, radius+2):
            if self.mask[center[0]+i, center[1]-radius-1] == 0:
                patch_border.append((center[0]+i, center[1]-radius-1))
            if self.mask[center[0]+i, center[1]+radius+1] == 0:
                patch_border.append((center[0]+i, center[1]+radius+1))
        
        for j in range(-radius-1, radius+2):
            if self.mask[center[0]-radius-1, center[1]+j] == 0:
                patch_border.append((center[0]-radius-1, center[1]+j))
            if self.mask[center[0]+radius+1, center[1]+j] == 0:
                patch_border.append((center[0]+radius+1, center[1]+j))

        contour = contour + patch_border
        self.contour = contour

        # Update mask
        only_patch = np.zeros(self.shape)
        only_patch = only_patch[:,:,0]
        only_patch[center[0]-radius:center[0]+radius+1, center[1]-radius:center[1]+radius+1] = 1
        self.mask = np.logical_or(self.mask, only_patch)

        # plot contour
        if plot:
            plt.imshow(self.arr)
            plt.plot([c[1] for c in contour], [c[0] for c in contour], 'r.')
            plt.scatter(center[1], center[0], c='b')
            plt.show()
            plt.imshow(self.mask)
            plt.show()

    def update_mask(self, patch, plot = False):
        center, radius = (patch.position, patch.radius)

        # Update mask
        only_patch = np.zeros(self.shape)
        only_patch = only_patch[:,:,0]
        only_patch[center[0]-radius:center[0]+radius+1, center[1]-radius:center[1]+radius+1] = 1
        self.mask = np.logical_or(self.mask, only_patch)

        # plot contour
        if plot:
            plt.imshow(self.mask)
            plt.show()
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
    
    def upsize_image(self, patch_size):
        self.prec_arr = self.arr
        print(self.arr.dtype)
        upsize = ((int(np.floor(self.arr.shape[0]/patch_size)+1)*patch_size), int((np.floor(self.arr.shape[1]/patch_size)+1)*patch_size), 3)
        print(upsize)
        self.arr = np.zeros(upsize, dtype=self.arr.dtype)
        self.arr[:self.prec_arr.shape[0], :self.prec_arr.shape[1], :] = self.prec_arr
        # Extend pixels
        for i in range(self.prec_arr.shape[0], self.arr.shape[0]):
            for j in range(self.prec_arr.shape[1]):
                self.arr[i,j,:] = self.arr[i-1,j,:]
        
        for i in range(self.prec_arr.shape[1], self.arr.shape[1]):
            for j in range(self.arr.shape[0]):
                self.arr[j,i,:] = self.arr[j,i-1,:]

        self.shape = self.arr.shape

        new_mask = np.ones((self.shape[0], self.shape[1]))
        new_mask[:self.prec_arr.shape[0], :self.prec_arr.shape[1]] = self.mask
        self.mask = new_mask

    def recrop_image(self):
        init_shape = self.prec_arr.shape
        self.arr = self.arr[:init_shape[0], :init_shape[1], :]
        self.shape = self.arr.shape

    def save_image(self, arr=None):
        if arr is None:
            arr = self.arr
        img = Image.fromarray(arr)
        nb_image = len(os.listdir("log_image"))//3
        img.save("log_image/result"+str(nb_image).zfill(3)+".jpg")

        plt.clf()
        plt.imshow(self.mask)
        plt.savefig("log_image/mask"+str(nb_image).zfill(3)+".jpg")

        contour = self.contour
        plt.clf()
        plt.imshow(arr)
        plt.plot([c[1] for c in contour], [c[0] for c in contour], 'r.')
        plt.savefig("log_image/contour"+str(nb_image).zfill(3)+".jpg")

    # ------------------------------------ PROPAGATING TEXTURE ------------------------

    def find_best_patch(self, patch, nb_patch = "default", method = "SSD"):
        # Return the best patch to replace the given one
        # patch : Patch
        # Return : Patch
        if nb_patch == "default":
            nb_patch = self.shape[0]*self.shape[1]//patch.radius**2

        best_patch = None
        best_distance = float("inf")
        data = patch.data
        if method == "MC":
            mean_color = np.sum(data, axis=(0,1), dtype=np.int32)
            patch_mask = self.mask[patch.position[0]-patch.radius:patch.position[0]+patch.radius+1, patch.position[1]-patch.radius:patch.position[1]+patch.radius+1]
            nb_one = np.sum(patch_mask)
            if nb_one != 0:  
                mean_color = np.floor(mean_color/nb_one).astype(np.uint8)
            else:
                # Get bigger patch
                data = self.arr[patch.position[0]-patch.radius-1:patch.position[0]+patch.radius+2, patch.position[1]-patch.radius-1:patch.position[1]+patch.radius+2]
                patch_mask = self.mask[patch.position[0]-patch.radius-1:patch.position[0]+patch.radius+2, patch.position[1]-patch.radius-1:patch.position[1]+patch.radius+2]
                nb_one = np.sum(patch_mask)
                if nb_one != 0:
                    mean_color = np.floor(np.sum(data, axis=(0,1), dtype=np.int32)/nb_one).astype(np.uint8)
                else:
                    mean_color = np.array([128,128,128])        
                    print("No one in patch")    
        random_center = np.random.randint((0+patch.radius, 0+patch.radius), (self.shape[0]-patch.radius, self.shape[1]-patch.radius), (nb_patch, 2))
        
        for k in range(nb_patch):
            p = Patch(data=self.arr[random_center[k,0]-patch.radius:random_center[k,0]+patch.radius+1, random_center[k,1]-patch.radius:random_center[k,1]+patch.radius+1], position=(random_center[k,0], random_center[k,1]), radius=patch.radius)
            if self.mask[p.position]: # Condition on confidence
                if method == "SSD":
                    distance = np.sum(np.square(data - p.data, dtype=np.int64))
                elif method == "MC": # Mean Color
                    distance = np.sum(np.square(mean_color-np.mean(p.data, axis=(0,1), dtype=np.int32), dtype=np.int32))
                
                if distance < best_distance:
                    best_distance = distance
                    best_patch = p
        if best_patch is None:
            print("No best patch found")
        else:
            return best_patch

    def propagate_texture(self, verbose = False):
        list_id_priority = np.array([[k, self.patches[k].priority] for k in range(len(self.patches))])
        list_id_priority = list_id_priority[list_id_priority[:,0].argsort()]

        if verbose:
            t1 = 0
            t2 = 0
            t3 = 0

        for index in range(len(self.patches)):
            i = int(list_id_priority[index,0])
            if self.patches[i].active:
                # Find best patch
                t = time.time()
                best_patch = self.find_best_patch(self.patches[i], method="MC")

                if verbose:
                    t1 += time.time()-t

                # Replace patch
                self.patches[i].data = best_patch.data
                self.patches[i].set_state(False)

                # Update mask
                t = time.time()
                self.update_mask(self.patches[i], plot=False)

                if verbose:
                    t2 += time.time()-t

                # Update image
                t = time.time()
                pos = self.patches[i].position
                radius = self.patches[i].radius
                hindex = (pos[0]-radius, pos[0]+radius)
                vindex = (pos[1]-radius, pos[1]+radius)
                self.arr[hindex[0]:hindex[1]+1, vindex[0]:vindex[1]+1] = self.patches[i].data
                if verbose:
                    t3 += time.time()-t

            else:
                #print("Patch " + str(self.patches[i].position) + " not active")
                pass
        if verbose:
            print("Time to find best patch : " + str(t1))
            print("Time to update contour : " + str(t2))
            print("Time to update image : " + str(t3))

    def update_priorities(self):
        for p in self.patches:
            p.update_priority(self.contour, self.mask)
    
    def get_active_patches(self, verbose = False):
        if verbose:
            print("Updating active patches")
            nb_active = 0

        for k in range(len(self.patches)):
            center, radius = self.patches[k].position, self.patches[k].radius
            for c in self.contour:
                if c[0]>=center[0]-radius and c[0]<=center[0]+radius and c[1]>=center[1]-radius and c[1]<=center[1]+radius:
                    self.patches[k].active = True
                    if verbose:
                        nb_active += 1
                    break
                else:
                    self.patches[k].active = False

        if verbose:
            print("Number of active patches : " + str(nb_active) + "/" + str(len(self.patches)))
            nb_active = 0
            for p in self.patches:
                if p.active:
                    nb_active += 1
            print("Number of active patches verif : " + str(nb_active) + "/" + str(len(self.patches)))

    #-------------------------- MAIN ----------------------------

    def main(self, image_path, mask_path, patch_size, verbose = False):
        self.load_image(image_path)
        self.load_mask(mask_path)
        self.find_contour(smoothing=True)

        # Remove mask
        uint8mask = self.mask.astype(np.uint8)
        self.arr = self.arr * uint8mask[:,:,np.newaxis]
        self.save_image()

        self.create_patches(patch_size)
        self.upsize_image(patch_size)

        bar = tqdm.tqdm()

        while len(self.contour) > 0:
            print("Contour length : " + str(len(self.contour)))
            bar.update(1)

            t = time.time()
            self.get_active_patches()
            if verbose:
                print("Active patches : " + str(time.time()-t))
                t = time.time()
            
            self.update_priorities()
            if verbose:
                print("Update priorities : " + str(time.time()-t))
                t = time.time()

            self.propagate_texture(verbose = verbose)
            self.find_contour(smoothing=True)

            self.save_image(self.arr)

            if verbose:
                print("Propagate texture : " + str(time.time()-t))
        
        self.recrop_image()
        self.print_image()


if __name__=="__main__":
    m = Main()
    m.main("image2.jpg", "mask2.ppm", 9)