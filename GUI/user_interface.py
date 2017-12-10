import Tkinter as tk
import tkFont as font
import os
import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)
import time
from threading import Thread
import Queue

X0=70
Y0=25
X1=230
Y1=285

HEIGHT=290

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set(self, x, y):
        self.x = x
        self.y = y


class RunThread(Thread):
    def __init__(self, queue0, queue1, model, learning_rate,epoch,batchsize,regulization):
        Thread.__init__(self)
        self.queue0 = queue0
        self.queue1 = queue1
        self.model = model
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batchsize = batchsize
        self.regulization = regulization
    
    def run(self):
        if self.model=='MNIST Fully Connected':

            # Layers and data
            layers = [
                DenseLayer(100,scale=2e-2),
                ReLu(alpha=0.01),
                DenseLayer(100,scale=2e-2),
                ReLu(alpha=0.01),
                DenseLayer(10,scale=2e-2),
                Softmax()
            ]
            x_t, y_t = get_mnist_data('../data/mnist/mnist_train.csv',50000)
            x_v, y_v = get_mnist_data('../data/mnist/mnist_test.csv',10000)

            # Let there be 500 points
            num_run = 500
            num_pass = len(x_t) * self.epoch / (self.batchsize * num_run)

            data = {
                'x_train': x_t,
                'y_train': y_t,
                'x_val':   x_v,
                'y_val':   y_v
            }
        elif self.model=='Cifar10 Fully Connected':
            layers = [
                DenseLayer(1000,scale=2e-2),
                ReLu(alpha=0.01),
                DenseLayer(500,scale=2e-2),
                ReLu(alpha=0.01),
                DenseLayer(250,scale=2e-2),
                ReLu(alpha=0.01),
                DenseLayer(10,scale=2e-2),
                Softmax()
            ]

            x_t, y_t = np.zeros((50000,3072)), np.zeros(50000)
            for i in xrange(5):
                x_sub, y_sub = get_cifar10_data('../data/cifar-10/data_batch_'+str(i+1),10000)
                x_t[10000*i : 10000*(i+1)] = x_sub.reshape(10000,3072)
                y_t[10000*i : 10000*(i+1)] = y_sub

            x_v, y_v = get_cifar10_data('../data/cifar-10/test_batch',10000)
            x_v = x_v.reshape(10000,3072)

            x_t /= 32.; x_v /= 32.

            data = {
                'x_train': x_t,
                'y_train': y_t,
                'x_val':   x_v,
                'y_val':   y_v
            }

            # Let there be 500 points
            num_run = 500
            num_pass = len(x_t) * self.epoch / (self.batchsize * num_run)


        else:

            # Let there be 500 points
            num_run = 500
            num_pass = len(x_t) * self.epoch / (self.batchsize * num_run)


            x_t, y_t = np.zeros((50000,3,32,32)), np.zeros(50000)
            for i in xrange(5):
                x_t[10000*i : 10000*(i+1)], y_t[10000*i : 10000*(i+1)]=get_cifar10_data('../data/cifar-10/data_batch_'+str(i+1),10000)
            x_v, y_v = get_cifar10_data('../data/cifar-10/test_batch',10000)
            
            data = {
                'x_train': x_t,
                'y_train': y_t,
                'x_val':   x_v,
                'y_val':   y_v
            }



            layers = [
                ConvLayer(3,(5,5),1,0),
                ReLu(),
                ConvLayer(3,(5,5),1,4),
                ReLu(),
                MaxPool((8,8),8),
                ConvLayer(3,(5,5),1,4),
                ReLu(),
                ConvLayer(3,(5,5),1,4),
                ReLu(),
                MaxPool((2,2),2),
                ConvLayer(3,(5,5),1,4),
                ReLu(),
                ConvLayer(1,(5,5),1,4),
                ReLu(),
                MaxPool((2,2),2),
                DenseLayer(10, scale=2e-2),
                Softmax()
            ]



        # Build the network
        args = {'learning_rate':self.learning_rate, 'epoch':self.epoch, 'batch_size':self.batchsize, 'reg':self.regulization, 'debug':0}
        fm = frame(layers, data, args)

        # Let there be 500 points
        num_run = 250
        num_pass = len(x_t) * self.epoch / (self.batchsize * num_run)


        for i in xrange(num_run):
            train_acc, val_acc, _, _ = fm.passes(num_pass = num_pass, val_num=10*self.batchsize)
            self.queue0.put(train_acc)
            self.queue1.put(val_acc)
            print("train_acc: "+str(train_acc))
            print("val_acc: "+str(val_acc))
            # a0_graph=self.update_graph(a0_graph, 0, train_acc)
            # a1_graph=self.update_graph(a1_graph, 1, val_acc)
            # TODO: do whatever necessary with those 2 values

class UI(tk.Tk):
    def __init__(self, master):
        tk.Tk.__init__(self)
        self.master = master
        self.models = ['MNIST Fully Connected', 'Cifar10 Fully Connected', 'Cifar10 CNN']
        self.data = ["mnist", "cifar10"]
        self.learning_rate = None
        self.epoch = 0
        self.batchsize = None
        self.regulization = None
        self.graphs = []
        self.queue0 = None
        self.queue1 = None
        # self.a0_graph = None
        # self.a1_graph = None
        self.prev_pt0 = Point(0,HEIGHT)
        self.prev_pt1 = Point(0,HEIGHT)
        
        self.title("Not Hotdog")
        self.geometry("800x640")
        self.resizable(False, False)

        self.left_frame = tk.Frame(self, width=300, height=800)
        self.left_frame.pack_propagate(0)
        self.right_canvas = tk.Canvas(self, width=500, height=800)
        self.right_canvas.pack_propagate(0)
        # configure option menu and text fields

        # frame title
        self.frame_title = tk.Label(self.left_frame, text="Config", bg="white", fg="black")

        # model dropdown menu
        self.model_var = tk.StringVar(self.left_frame)
        self.model_var.set(self.models[0])
        self.model_dropdown = tk.OptionMenu(self.left_frame, self.model_var, *self.models)

        # choose data sets

        # self.data_var = tk.StringVar(self.left_frame)
        # self.data_var.set(self.data[0])
        # self.data_dropdown = tk.OptionMenu(self.left_frame, self.data_var, *self.data)


        # choose learning rate

        self.internal_frame = tk.Frame(self.left_frame, width=300, height=400)

        tk.Label(self.internal_frame, text="Learning rate:").grid(row=0, pady = 20)
        tk.Label(self.internal_frame, text="Epoch:").grid(row=1)
        tk.Label(self.internal_frame, text="Batchsize:").grid(row=2, pady = 20)
        tk.Label(self.internal_frame, text="Regularization:").grid(row=3)


        self.e1 = tk.Entry(self.internal_frame)
        self.e2 = tk.Entry(self.internal_frame)
        self.e3 = tk.Entry(self.internal_frame)
        self.e4 = tk.Entry(self.internal_frame)

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=2, column=1)
        self.e4.grid(row=3, column=1)

        #buttons
        
        tk.Button(self.internal_frame, text='Start', command=self.start_training).grid(row=4, column=0, sticky=tk.W,padx=10, pady=20)
        tk.Button(self.internal_frame, text = 'Clear', command =self.magic_clear).grid(row=5,column = 0, sticky = tk.W, padx=10,pady=20)
        tk.Button(self.internal_frame, text='Quit', command=self.magic_exit).grid(row=4, column=1, sticky=tk.E, pady=20)

        self.frame_title.pack(side=tk.TOP, fill=tk.X)
        self.model_dropdown.pack(side=tk.TOP,fill=tk.X, pady = 40)
        # self.data_dropdown.pack(side=tk.TOP,fill=tk.X)
        self.internal_frame.pack(side=tk.TOP, fill=tk.X, pady = 40)


        # set up canvases
        self.acc1_title = tk.Label(self.right_canvas, text="Training Accuracy", background="grey", foreground="white")
        acc1_space = tk.Canvas(self.right_canvas, background="white", width=500, height=290)
        acc1_space.pack_propagate(0)

        # acc1_space.create_rectangle(70, 25, 230, 285, fill="blue")

        self.acc2_title = tk.Label(self.right_canvas, text="Validation Accuracy", background="grey", foreground="white")
        acc2_space = tk.Canvas(self.right_canvas, background="white", width=500, height=290)
        acc2_space.pack_propagate(0)

        self.acc1_title.pack(side=tk.TOP, fill=tk.X)
        acc1_space.pack(side=tk.TOP)
        
        self.acc2_title.pack(side=tk.TOP, fill=tk.X)
        acc2_space.pack(side=tk.BOTTOM)
        
        self.graphs.append(acc1_space)
        self.graphs.append(acc2_space)

        self.right_canvas.pack(side=tk.RIGHT)
        self.left_frame.pack(side=tk.LEFT)

    def start_training(self):
        model = self.model_var.get()
        # data = self.data_var.get()

        print("current model: "+model)
        print("learning_rate: "+self.e1.get())
        print("epoch: "+self.e2.get())
        print("batchsize: "+ self.e3.get())
        print("regularization: "+ self.e4.get())
        self.learning_rate = float(self.e1.get())
        self.epoch = int(self.e2.get())
        self.batchsize = int(self.e3.get())
        self.regulization = float(self.e4.get())

        # self.a0_graph = self.graphs[0].create_rectangle(X0, Y0, X1, Y1, fill="blue")
        # self.a1_graph = self.graphs[1].create_rectangle(X0, Y0, X1, Y1, fill="red")

        self.queue0 = Queue.Queue()
        self.queue1 = Queue.Queue()

        RunThread(self.queue0, self.queue1, model,self.learning_rate,self.epoch,self.batchsize,self.regulization).start()
        self.master.after(100, self.update_graph)


    def update_graph(self):
        
        try:
            acc0 = self.queue0.get(0)
            acc1 = self.queue1.get(0)
            canvas0 = self.graphs[0]
            canvas1 = self.graphs[1]
            newPt0 = Point(self.prev_pt0.x+2, int(HEIGHT-HEIGHT*acc0))
            newPt1 = Point(self.prev_pt1.x+2, int(HEIGHT-HEIGHT*acc1))
            canvas0.create_line(self.prev_pt0.x, self.prev_pt0.y, newPt0.x, newPt0.y, fill="blue", width=3)
            canvas1.create_line(self.prev_pt1.x, self.prev_pt1.y, newPt1.x, newPt1.y, fill="red", width=3)
            self.acc1_title.config(text = "Training Accuracy: "+str(acc0))
            self.acc2_title.config(text = "Validation Accuracy:  "+str(acc1))
            self.prev_pt0 = newPt0
            self.prev_pt1 = newPt1
            # height = Y1 - Y0
            # newY0_0 = int(height*acc0+Y0)
            # newY0_1 = int(height*acc1+Y0)
            # canvas0.delete(self.a0_graph)
            # canvas1.delete(self.a1_graph)
            # self.a0_graph = canvas0.create_rectangle(X0, newY0_0, X1, Y1, fill="blue")
            # self.a1_graph = canvas1.create_rectangle(X0, newY0_1, X1, Y1, fill="red")
            # print("new acc0: "+ str(newY0_0-Y0))
            # print("new acc1: "+ str(newY0_1-Y0))
            self.update_graph()

        except Queue.Empty:
            self.master.after(100, self.update_graph)
        
    def magic_exit(self):
        exit(0)

    def magic_clear(self):
        canvas0 = self.graphs[0]
        canvas1 = self.graphs[1]
        canvas0.delete("all")
        canvas1.delete("all")
        self.prev_pt0 = Point(0,HEIGHT)
        self.prev_pt1 = Point(0,HEIGHT)
        self.acc1_title.config(text = "Training Accuracy")
        self.acc2_title.config(text = "Validation Accuracy")

if __name__ == "__main__":
    root = tk.Tk()
    ui = UI(root)
    ui.mainloop()
