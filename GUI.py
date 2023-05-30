import ttkbootstrap as ttk
import tkinter as tk
from ttkbootstrap.constants import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image

import os

# Create a GUI class containing an image and rolling menu
class GUI:
    def __init__(self) -> None:
        # Initiale Values
        self.filename = None

        self.root = ttk.Window()
        self.root.title("InPainting by Clément & Thomas")
        self.root.geometry("800x600")
        self.open_file()

        # Add the image
        self.print_image()

    def open_file(self):
        filetype = (("Images File", "*.jpg *.png *.jpeg *.gif"), 
                    ("All Files", "*.*"))

        initialdir = os.path.dirname(os.path.realpath(__file__))

        self.filename = fd.askopenfilename(
            title="Open a file",
            initialdir=initialdir,
            filetypes=filetype
        )

    def print_image(self):
        if self.filename is None:
            print("No file selected")
            return
        img = Image.open(self.filename)
        img = img.resize((400, 400), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(img)
        self.image_label = ttk.Label(self.root, image=self.image)
        self.image_label.pack()



if __name__ == "__main__":
    gui = GUI()
    gui.root.mainloop()