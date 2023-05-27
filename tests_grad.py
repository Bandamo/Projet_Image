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


    selected_patch = m.patches[959]

    # Convert to Lab and next to grayscale
    selected_patch.data = cv2.cvtColor(selected_patch.data, cv2.COLOR_BGR2LAB)
    selected_patch.data = cv2.cvtColor(selected_patch.data, cv2.COLOR_BGR2GRAY)
    selected_patch.data = cv2.GaussianBlur(selected_patch.data, (7,7), 0)

    plt.figure()
    plt.imshow(selected_patch.data)
    # Compute the gradient of the patch and smooth it
    grad_x = np.gradient(selected_patch.data)[0]
    grad_y = np.gradient(selected_patch.data)[1]

    # Outline a patch on the image
    plt.figure()
    plt.imshow(m.image)
    plt.plot([selected_patch.position[1] - selected_patch.radius, selected_patch.position[1] - selected_patch.radius], [selected_patch.position[0] - selected_patch.radius, selected_patch.position[0] + selected_patch.radius], color='red')
    plt.plot([selected_patch.position[1] + selected_patch.radius, selected_patch.position[1] + selected_patch.radius], [selected_patch.position[0] - selected_patch.radius, selected_patch.position[0] + selected_patch.radius], color='red')
    plt.plot([selected_patch.position[1] - selected_patch.radius, selected_patch.position[1] + selected_patch.radius], [selected_patch.position[0] - selected_patch.radius, selected_patch.position[0] - selected_patch.radius], color='red')
    plt.plot([selected_patch.position[1] - selected_patch.radius, selected_patch.position[1] + selected_patch.radius], [selected_patch.position[0] + selected_patch.radius, selected_patch.position[0] + selected_patch.radius], color='red')

    # Draw the contour on the image
    plt.plot([c[1] for c in m.contour], [c[0] for c in m.contour], 'g.')

    m.contour = np.array(m.contour)
    selected_patch.position = np.array(selected_patch.position)

    closest_pix = selected_patch.get_closest_pixel(m.contour, selected_patch.position)

    plt.plot(closest_pix[1], closest_pix[0],marker='*', color='blue')

    print(selected_patch.perpendicular_vector((grad_x[closest_pix[0]-selected_patch.position - 4, closest_pix[1]-selected_patch.position - 4], grad_y[closest_pix[0]-selected_patch.position -4, closest_pix[1]-selected_patch.position-4])))

    # Draw a vector field and invert the y axis
    plt.figure()
    plt.quiver(grad_y, grad_x)
    plt.gca().invert_yaxis()
    

    plt.show()