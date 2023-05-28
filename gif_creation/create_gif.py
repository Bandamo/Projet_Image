import os
from PIL import Image

def create_gif(image_list, gif_name, duration = 500):
    '''
    Create a gif from a list of images
    '''
    frames = []
    for image_name in image_list:
        frames.append(Image.open(image_name))
    
    # Save into a GIF file that loops forever
    frames[0].save(gif_name, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration, loop=0)
    
    return

if __name__=="__main__":
    l = os.listdir("gif_creation/images")
    l = ["gif_creation/images/" + i for i in l]
    l.sort()
    create_gif(l, "gif_creation/patch.gif")
