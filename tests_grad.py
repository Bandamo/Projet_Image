import numpy as np
import scipy as sc
from patch import Patch
from main import Main
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    m = Main()
    m.load_image("image.jpg")
    m.load_mask("mask.ppm")
    m.find_contour(plot = False)
    m.create_patches(9)
            
    selected_patch = m.patches[1254] #959 1254

    # Convert to grayscale
    selected_patch.data = cv2.cvtColor(selected_patch.data, cv2.COLOR_BGR2GRAY)
    selected_patch.data = cv2.GaussianBlur(selected_patch.data, (7,7), 0)

    # Compute the gradient of the patch and smooth it
    grad_x = np.gradient(selected_patch.data)[0]
    grad_y = np.gradient(selected_patch.data)[1]

    m.contour = np.array(m.contour)
    selected_patch.position = np.array(selected_patch.position)


    # Get the coordinates of the biggest gradient
    max_coord = np.unravel_index(np.argmax(np.sqrt(grad_x**2 + grad_y**2)), grad_x.shape)
    max_grad = np.array([grad_x[max_coord], grad_y[max_coord]])
    print("Max grad: ", max_grad)

    closest_pix_org = selected_patch.get_closest_pixel(m.mask, selected_patch.position)
    closest_pix = closest_pix_org.copy()

    # Change the coordinates to the patch coordinates
    closest_pix[0] = closest_pix_org[0] - selected_patch.position[0] + selected_patch.radius
    closest_pix[1] = closest_pix_org[1] - selected_patch.position[1] + selected_patch.radius
    print(closest_pix)

    #isophote_method_1 = selected_patch.perpendicular_vector([grad_x[closest_pix[0], closest_pix[1]], grad_y[closest_pix[0], closest_pix[1]]])
    isophote_method_2 = selected_patch.perpendicular_vector(max_grad)
    #print("Isophote method 1: ", isophote_method_1)
    print("Isophote method 2: ", isophote_method_2)

    # Draw a vector field and invert the y axis
    plt.figure()
    Q = plt.quiver(grad_y, grad_x)
    #plt.quiver(closest_pix[1], closest_pix[0], isophote_method_1[1], isophote_method_1[0], color='red', scale=Q.scale)
    plt.quiver(max_coord[1], max_coord[0], isophote_method_2[1], isophote_method_2[0], color='green', scale=Q.scale)
    plt.gca().invert_yaxis()
    #plt.plot(closest_pix[1], closest_pix[0], 'b*')
    
    plt.figure()
    plt.imshow(selected_patch.data)
    #plt.plot(closest_pix[1], closest_pix[0], 'b*')

    # Outline a patch on the image
    plt.figure()
    plt.imshow(m.image)
    plt.plot([selected_patch.position[1] - selected_patch.radius, selected_patch.position[1] - selected_patch.radius], [selected_patch.position[0] - selected_patch.radius, selected_patch.position[0] + selected_patch.radius], color='red')
    plt.plot([selected_patch.position[1] + selected_patch.radius, selected_patch.position[1] + selected_patch.radius], [selected_patch.position[0] - selected_patch.radius, selected_patch.position[0] + selected_patch.radius], color='red')
    plt.plot([selected_patch.position[1] - selected_patch.radius, selected_patch.position[1] + selected_patch.radius], [selected_patch.position[0] - selected_patch.radius, selected_patch.position[0] - selected_patch.radius], color='red')
    plt.plot([selected_patch.position[1] - selected_patch.radius, selected_patch.position[1] + selected_patch.radius], [selected_patch.position[0] + selected_patch.radius, selected_patch.position[0] + selected_patch.radius], color='red')

    # Draw the contour on the image
    plt.plot([c[1] for c in m.contour], [c[0] for c in m.contour], 'g.')
    plt.plot(closest_pix_org[1], closest_pix_org[0],marker='*', color='blue')

    plt.figure()
    plt.plot(np.gradient(m.mask[selected_patch.position[1] - selected_patch.radius:selected_patch.position[1] + selected_patch.radius, selected_patch.position[0] - selected_patch.radius:selected_patch.position[0] + selected_patch.radius]))

    plt.show()