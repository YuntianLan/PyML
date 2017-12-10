import Tkinter as tk
import tkFont as font
import os
import numpy as np
import sys
sys.path.append('..')
from core.frame import *
# from data.data_util import *
# sys.path.pop(-1)


class UI(tk.Tk):
    def __init__(self, models):
        tk.Tk.__init__(self)

        self.models = models
        self.data = ["mnist", "cifar10"]
        self.learning_rate = None
        self.epoch = 0
        self.batchsize = None
        self.regulization = None
        self.graphs = []
        
        self.title("Not Hotdog")
        self.geometry("800x640")
        self.resizable(False, False)

        self.left_frame = tk.Frame(self, width=300, height=800)
        self.left_frame.pack_propagate(0)
        self.right_canvas = tk.Canvas(self, width=500, height=800)
        self.right_canvas.pack_propagate(0)
        # configure option menu and text fields

  		# frame title
        self.frame_title = tk.Label(self.left_frame, text="Title", bg="white", fg="black")

        # model dropdown menu
        self.model_var = tk.StringVar(self.left_frame)
        self.model_var.set(self.models[0])
        self.model_dropdown = tk.OptionMenu(self.left_frame, self.model_var, *self.models)

        # choose data sets

        self.data_var = tk.StringVar(self.left_frame)
        self.data_var.set(self.data[0])
        self.data_dropdown = tk.OptionMenu(self.left_frame, self.data_var, *self.data)


        # choose learning rate

        self.internal_frame = tk.Frame(self.left_frame, width=300, height=400)

        tk.Label(self.internal_frame, text="Learning rate:").grid(row=0, pady = 20)
        tk.Label(self.internal_frame, text="Epoch").grid(row=1)
        tk.Label(self.internal_frame, text="Batchsize:").grid(row=2, pady = 20)
        tk.Label(self.internal_frame, text="Regulization:").grid(row=3)


        self.e1 = tk.Entry(self.internal_frame)
        self.e2 = tk.Entry(self.internal_frame)
        self.e3 = tk.Entry(self.internal_frame)
        self.e4 = tk.Entry(self.internal_frame)

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=2, column=1)
        self.e4.grid(row=3, column=1)

        #buttons
        
        tk.Button(self.internal_frame, text='Start', command=self.start_training).grid(row=4, column=0, sticky=tk.W,padx=20, pady=20)
        tk.Button(self.internal_frame, text='Quit', command=self.internal_frame.quit).grid(row=4, column=1, sticky=tk.E, pady=20)

        self.frame_title.pack(side=tk.TOP, fill=tk.X)
        self.model_dropdown.pack(side=tk.TOP,fill=tk.X, pady = 40)
        self.data_dropdown.pack(side=tk.TOP,fill=tk.X)
        self.internal_frame.pack(side=tk.TOP, fill=tk.X, pady = 40)

        acc1_title = tk.Label(self.right_canvas, text="Accuracy1", background="grey", foreground="white")
        acc1_space = tk.Canvas(self.right_canvas, background="white", width=300, height=290)
        acc1_space.pack_propagate(0)

        acc2_title = tk.Label(self.right_canvas, text="Accuracy2", background="grey", foreground="white")
        acc2_space = tk.Canvas(self.right_canvas, background="white", width=300, height=290)
        acc2_space.pack_propagate(0)

        acc1_title.pack(side=tk.TOP, fill=tk.X)
        acc1_space.pack(side=tk.TOP)
        
        acc2_title.pack(side=tk.TOP, fill=tk.X)
        acc2_space.pack(side=tk.BOTTOM)
        
        self.graphs.append(acc1_space)
        self.graphs.append(acc2_space)

        self.right_canvas.pack(side=tk.RIGHT)
        self.left_frame.pack(side=tk.LEFT)

    def start_training(self):
    	model = self.model_var.get()
    	data = self.data_var.get()
        self.learning_rate = self.e1.get()
        self.epoch = self.e2.get()
        self.batchsize = self.e3.get()
        self.regulization = self.e4.get()
        print("current model: "+model)
        print("current data: "+data)
        print("learning_rate: "+self.learning_rate)
        print("epoch: "+self.epoch)
        print("batchsize: "+ self.batchsize)
        print("regulization: "+ self.regulization)

    def update_graph(self):
    	pass



if __name__ == "__main__":
    models = ["model1", "model2", "model3", "model4"]
    ui = UI(models)
    ui.mainloop()
