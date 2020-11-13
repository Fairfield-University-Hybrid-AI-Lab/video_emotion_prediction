import pandas as pd
import numpy as np
import boundingbox
import glob

'''
calculate the quantiles from annotations in the annnotiation files
'''
ANNOTATIONS_PATH =  ".\\annotations"
file_paths = glob.glob(".\\annotations\\valence\\*.txt")
valence_annotations = []
for file_path in file_paths:
    for annotation in boundingbox.load_annotations(file_path):
        valence_annotations.append(annotation)

file_paths = glob.glob(".\\annotations\\arousal\\*.txt")
arousal_annotations = []
for file_path in file_paths:
    for annotation in boundingbox.load_annotations(file_path):
        arousal_annotations.append(annotation)

valence_annotations = [float(i) for i in valence_annotations]
arousal_annotations = [float(i) for i in arousal_annotations]

#valence class edges
print(np.quantile(valence_annotations, .33)) 
print(np.quantile(valence_annotations, .66))

#arousal class edges
print(np.quantile(arousal_annotations, .33)) 
print(np.quantile(arousal_annotations, .66))