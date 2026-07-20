import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import imageio
from IPython.display import Video
from IPython.display import display
from PIL import Image, ImageDraw

# create TF dataset
dataset_name = "mg_panda_flip_mug"
print(f"Loading data from dataset: {dataset_name}")
# module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, data_dir="/iliad/u/jenseng/tensorflow_datasets/", split='train')

breakpoint()