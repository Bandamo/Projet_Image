import ttkbootstrap as ttkb
import tkinter as tk
from ttkbootstrap.constants import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import io
from PIL import ImageGrab

import os

class MainToolbar(ttkb.Frame):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.__parent = parent
        self.__imgselect = ttkb.PhotoImage(file="images/select_icon.png").subsample(22)
        self.select_tool = ttkb.Button(self, text='Paint', compound=ttkb.LEFT, image=self.__imgselect, command=self.select)
        self.select_tool.grid(row=0, column=0)
        self.select_tool.config(style="success.TButton")
        self.__imgdelete = ttkb.PhotoImage(file="images/delete_icon.png").subsample(22)
        self.delete_tool = ttkb.Button(self, text='Erase', compound=ttkb.LEFT, image=self.__imgdelete, command=self.erase)
        self.delete_tool.grid(row=0, column=3)
        self.delete_tool.config(style="danger.TButton")
        self.__imgeraseall = ttkb.PhotoImage(file="images/delete_icon.png").subsample(22)
        self.erase_all_tool = ttkb.Button(self, text='Erase all', compound=ttkb.LEFT, image=self.__imgeraseall, command=self.erase_all)
        self.erase_all_tool.grid(row=0, column=4)
        self.erase_all_tool.config(style="danger.TButton")
        label = ttkb.Label(self, text="Brush size:")
        label.grid(row=0, column=5)
        self.brush_size = ttkb.Scale(self, from_=1, to=100, command=self.retreive_size)
        self.brush_size.grid(row=0, column=6)
        self.brush_size.set(15)



        self.__buttons = [self.select_tool, self.delete_tool]

        #self.selection_rectangle = None

    def clear_selection(self) -> None:
        """
        Permet de déselectionner tous les boutons de la barre d'outils
        """
        # for button in self.__buttons:
        #     button.config(relief=ttkb.RAISED)
        self.__parent.main_canvas.unbind("<ButtonRelease-1>")
        self.__parent.main_canvas.unbind("<ButtonPress-1>")

    def select(self) -> None:
        self.clear_selection()
        #self.__parent.maincanvas.bind("<ButtonPress-1>", self.create_selection_rectangle)
        #self.__parent.maincanvas.bind("<ButtonRelease-1>", self.delete_selection_rectangle)
        #self.__parent.maincanvas.tag_bind("node","<ButtonRelease-1>",self.__parent.maincanvas.node_left_cliked)
        #self.__parent.maincanvas.tag_bind("edge","<ButtonRelease-1>",self.__parent.maincanvas.edge_left_cliked)
        self.__parent.main_canvas.bind("<ButtonPress-1>", self.__parent.main_canvas.start_paint)
        self.__parent.main_canvas.bind("<ButtonRelease-1>", self.__parent.main_canvas.stop_paint)
    
    def erase(self) -> None:
        self.clear_selection()
        self.__parent.main_canvas.bind("<ButtonPress-1>", self.__parent.main_canvas.start_erase)
        self.__parent.main_canvas.bind("<ButtonRelease-1>", self.__parent.main_canvas.stop_erase)
    
    def erase_all(self) -> None:
        self.__parent.main_canvas.delete("paint")
        self.__parent.main_canvas.delete("erase")
    
    def retreive_size(self, event) -> None:
        self.__parent.main_canvas.set_brush_size(self.brush_size.get())

class Main_menu_bar(ttkb.Menu):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.file_menu = ttkb.Menu(self, tearoff=False)
        self.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_command(label="Save", command=self.save_mask)
        self.file_menu.add_command(label="Exit", command=self.quit)
    
    def open_file(self):
        filetype = (("Images File", "*.jpg *.png *.jpeg *.gif"), 
                    ("All Files", "*.*"))

        initialdir = os.path.dirname(os.path.realpath(__file__))

        self.parent.filename = fd.askopenfilename(
            title="Open a file",
            initialdir=initialdir,
            filetypes=filetype
        )
        self.parent.main_canvas.load_image(self.parent.filename)
    
    def save_mask(self):
        filetype = (("Images File", "*.jpg *.png *.jpeg *.gif"), 
                    ("All Files", "*.*"))

        initialdir = os.path.dirname(os.path.realpath(__file__))

        filename = fd.asksaveasfilename(
            title="Save mask",
            initialdir=initialdir,
            filetypes=filetype
        )
        self.parent.main_canvas.delete("image")
        self.parent.main_canvas.postscript(file = 'test.eps') 
        # use PIL to convert to PNG 
        img = Image.open('test.eps')
        img.convert()
        img.save(filename) 
        

class Main_Canvas(ttkb.Canvas):
    def __init__(self, parent):
        super().__init__(parent, bg='red') 
        self.parent = parent
        self.brush_size = 15
        self.image = Image.open("images/image.jpg")
        self.image_ratio = self.image.size[1] / self.image.size[0]
        self.resized_image_img = ImageTk.PhotoImage(self.image.resize((int(int(self.cget('width'))), int(int(self.cget('width'))*self.image_ratio))))
        self.canvas_img = self.create_image(int(int(self.cget('width')))/2,
                                            int(int(self.cget('width')))*self.image_ratio/2,
                                            image=self.resized_image_img,
                                            anchor=tk.CENTER,
                                            tags="image")

        self.bind("<Configure>", self.resize_callback)
    
    def resize_callback(self, *args):
        if int(int(self.winfo_width()))*self.image_ratio <= int(int(self.winfo_height())):
            self.resized_image_img = ImageTk.PhotoImage(self.image.resize((int(int(self.winfo_width())), int(int(self.winfo_width())*self.image_ratio))))
            self.coords(self.canvas_img, int(int(self.winfo_width())/2), int(int(self.winfo_width())*self.image_ratio/2))
        else:
            self.resized_image_img = ImageTk.PhotoImage(self.image.resize((int(int(self.winfo_height())/self.image_ratio), int(int(self.winfo_height())))))
            self.coords(self.canvas_img, int(int(self.winfo_height())/self.image_ratio/2), int(int(self.winfo_height())/2))
        self.itemconfigure(self.canvas_img, image=self.resized_image_img)

    def load_image(self, filename):
        self.image = Image.open(filename)
        self.image_ratio = self.image.size[1] / self.image.size[0]
        self.resized_image_img = ImageTk.PhotoImage(self.image.resize((int(int(self.cget('width'))), int(int(self.cget('width'))*self.image_ratio))))
        self.itemconfigure(self.canvas_img, image=self.resized_image_img)
    
    def start_paint(self, event):
        self.old_x = event.x
        self.old_y = event.y
        self.create_oval(event.x - self.brush_size, event.y - self.brush_size, event.x + self.brush_size, event.y + self.brush_size, fill='black', tags="paint")
        self.bind("<B1-Motion>", self.paint)
    
    def stop_paint(self, event):
        self.unbind("<B1-Motion>")

    def paint(self, event):
        self.create_oval(event.x - self.brush_size, event.y - self.brush_size, event.x + self.brush_size, event.y + self.brush_size, fill='black', tags="paint")
        self.old_x = event.x
        self.old_y = event.y
    
    def start_erase(self, event):
        self.old_x = event.x
        self.old_y = event.y
        self.create_oval(event.x - self.brush_size, event.y - self.brush_size, event.x + self.brush_size, event.y + self.brush_size, fill='white', tags="erase")
        self.bind("<B1-Motion>", self.erase)
    
    def stop_erase(self, event):
        self.unbind("<B1-Motion>")

    def erase(self, event):
        self.create_oval(event.x - self.brush_size, event.y - self.brush_size, event.x + self.brush_size, event.y + self.brush_size, fill='white', tags="erase")
        self.old_x = event.x
        self.old_y = event.y

    def set_brush_size(self, size):
        self.brush_size = size
    


class Main_Application(ttkb.Window):
    def __init__(self):
        super().__init__(title="InPainting by Clément & Thomas")
        self.main_canvas = Main_Canvas(self)
        self.main_toolbar = MainToolbar(self)
        self.main_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.main_canvas.pack(expand=True, fill=tk.BOTH, anchor=tk.CENTER)

        self.main_menu_bar = Main_menu_bar(self)
        self.config(menu=self.main_menu_bar)
        

if __name__ == "__main__":
    app = Main_Application()
    app.mainloop()