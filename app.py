import gradio as gr
import os
from functions import *

title = "DeepLense CvT: Substructure Prediction"
examples_dir = 'examples'

# loading the example files
examples = []
for root, dirs, files in os.walk(examples_dir):
    for file in files:
        if file.endswith(".jpg"):
            examples.append([os.path.join(root, file), os.path.join(root, file)])

# initialising the interface
interface = gr.Interface(fn=predict, inputs=gr.Image(type= 'pil', shape=(256, 256), image_mode= 'L'),
            outputs= gr.Label(), cache_examples= False,
            examples= examples, title= title, css= '.gr-box {background-color: rgb(230 230 230);}')

interface.launch()