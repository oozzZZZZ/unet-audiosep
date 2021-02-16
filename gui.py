#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:11:06 2021

@author: zankyo
"""

import tkinter as tk
from tkinter import filedialog
import os
import myutils as ut

# トップレベルウィンドウ作成
root = tk.Tk()
root.title("Separater")

#file
val_filename = tk.StringVar(root)
var_scale = tk.DoubleVar()
val_epoch = tk.StringVar(root)
val_outputdir = tk.StringVar(root)

def file_select():
    fTyp = [("","*.wav")]   
    filename = filedialog.askopenfilename(filetypes = fTyp,initialdir='~/')
    val_filename.set(filename)
    label1.config(text = str(val_filename.get()))
    
def dir_select():
    fld = filedialog.askdirectory(initialdir = '~/')
    val_outputdir.set(fld)
    label2.config(text = str(val_outputdir.get()))
                  
                  
def sel():
   selection = "Mask Rate = " + str(var_scale.get())+"%"
   label0.config(text = selection)
   
titlefont=('Helvetica', 15, "bold")

#wav
labeltitle1 = tk.Label(text='Select Wav File', font=titlefont)
labeltitle1 .pack(anchor="w")
button1 = tk.Button(root, text="Choose File", command=file_select)
button1.pack()
label1 = tk.Label(root, font=('Helvetica', 12), fg='gray')
label1.pack()

#output
labeltitle2 = tk.Label(text='Select Output Directory', font=titlefont)
labeltitle2 .pack(anchor="w")
button2 = tk.Button(root, text="Choose Output Directory", command=dir_select)
button2.pack()
label2 = tk.Label(root, font=('Helvetica', 12), fg='gray')
label2.pack()

#model
labeltitle3 = tk.Label(text='Select model', font=titlefont)
labeltitle3 .pack(anchor="w")
OptionList = ["epoch40","epoch50","epoch60","epoch80","epoch110","epoch250",]
val_epoch.set(OptionList[0])
opt = tk.OptionMenu(root, val_epoch, *OptionList)
opt.config(width=10, font=('Helvetica', 12))
opt.pack()
labelTest = tk.Label(text="", font=('Helvetica', 12), fg='gray')
labelTest.pack()
def callback(*args):
    labelTest.configure(text="The selected model is {}".format(val_epoch.get()))
val_epoch.trace("w", callback)

#Scale Title
label_maskrate = tk.Label(text='Hard Mask', font=titlefont)
label_maskrate.pack(anchor="w")

# tk.Scale

var_scale.set(30)
scale = tk.Scale(
    root,
    variable=var_scale,
    showvalue=True,
    orient=tk.HORIZONTAL,
    cursor="arrow",
    tickinterval=10,
    from_=50,
    to=100,
    length=500,
)

scale.pack(padx=10, pady=10)

button = tk.Button(root, text="Set Mask Rate", command=sel)
button.pack()
label0 = tk.Label(root, font=('Helvetica', 12), fg='gray')
label0.pack()

def start():
    label3.config(text = "")
    label4.config(text = "")
    filepath=val_filename.get()
    outputpath=val_outputdir.get()
    epoch_num=val_epoch.get()
    scale_num=var_scale.get()
    maskrate=int(scale_num)/100
    
    basename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
    
    outname_vocal = outputpath+"/"+basename_without_ext+"_vocal.wav"
    outname_inst = outputpath+"/"+basename_without_ext+"_inst.wav"
    outpath = "Vocal >> {}\nInst >> {}".format(outname_vocal,outname_inst)
    label3.config(text = outpath)
    
    ut.separation_main(filepath,epoch_num,maskrate,outname_vocal,outname_inst)
    label4.config(text = "Completed!")
    
    
    
button1 = tk.Button(root, text="Start Separation", font=("Helvetica", "16", "bold"),width=50,bg="AliceBlue",command=start)
button1.pack()    

label3 = tk.Label(root, font=('Helvetica', 12), fg='gray')
label3.pack()

label4 = tk.Label(root, font=("Helvetica", "16", "bold"))
label4.pack()

root.mainloop()