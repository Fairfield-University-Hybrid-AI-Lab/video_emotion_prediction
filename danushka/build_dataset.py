from os import makedirs
from os import listdir
from os import walk
from os import path
from shutil import copyfile
from random import seed
from random import random
'''
put the images into predefined training and test directories representing the labels. 
This is useful for model training in Tensorflow using flowfromdirectory
'''
dataset_home = 'dataset_valence/'
val_ratio = 0.25
src_directory = 'faces'
def main():
    # create directories
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['high/', 'neutral/', 'low/']
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)
    # seed random number generator
    seed(1)

  
    # copy training dataset images into subdirectories
    for dir in next(walk(src_directory))[1]:
        src = src_directory + '/'+ dir + '/'+'valence'+'/' 
        copy_to_dataset(src, 'high/', dir)
        copy_to_dataset(src, 'neutral/', dir)
        copy_to_dataset(src, 'low/', dir)
    
def copy_to_dataset(src, label, dir):
    if not path.exists(src+'/'+label):
        return
    for file in listdir(src + '/' +label):       
        print(src)
        if random() < val_ratio:
            dst_dir = 'test/'
            dst = dataset_home + dst_dir + label  + dir + file
            copyfile(src+label+file, dst)
        else:
            dst_dir = 'train/'
            dst = dataset_home + dst_dir + label  +dir+ file
            copyfile(src+ label +file, dst)

if __name__ == "__main__":
    main()