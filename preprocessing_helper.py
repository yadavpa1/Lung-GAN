#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:47:21 2021

@author: Lung-GANs
"""

import os
import sys
import os.path
from PIL import Image

import numpy as np
import splitfolders
import argparse

def split_name(file_name):
    return file_name.rsplit('.',1)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)
        
def convert_img_type(base_directory, path):
    file_path =  base_directory + '/' + path
    new_file_name = ""
    im = Image.open(file_path)
    new_file_name = split_name(path)[0] + ".jpg"
    new_file_path = base_directory + '/' + new_file_name
    
    im.convert("RGB").save(new_file_path)
    im.close()
    os.remove(file_path)
    return new_file_name

def convert_to_jpg(base_directory):
    print ("In progress: File type conversion for folder " + base_directory)
    for path in os.listdir(base_directory):
    
        old_name = base_directory + '/' + path
        
        #convert file to jpg format
        if os.path.isfile(old_name):
            if(split_name(path)[1]).lower() in ['jpg', 'jpeg', 'png', 'jfif']:
                if(split_name(path)[1] != "jpg"):
                    path = convert_img_type(base_directory, path)
            else:
                print("Deleting file " + old_name)
                os.remove(old_name)
        
        else:
           #If it's a subfolder, recursively call rename files on that directory.
           print("Recursing: " + old_name)
           convert_to_jpg(old_name)
    print ("Done: File type conversion for folder " + base_directory)

        
        
def rename_files(base_directory, num):

    print("In progress: File renaming for folder " + base_directory)
    #Get the folder name for base_directory (c:\users\foobar => foobar)
    directory_name = os.path.basename(base_directory)
    #List the files in base_directory
    for path in os.listdir(base_directory):
    
        old_name = base_directory + '/' + path
        
        #If the path points to a file, rename it directory name + '.' + extension
        if os.path.isfile(old_name):
            new_name = base_directory + '/' + directory_name + '_' + str(num) + '.' + split_name(path)[1]
            if not os.path.exists(new_name):
                os.rename(old_name,new_name)
                num = num + 1
            else:
                print("ERROR:"+new_name+" exists")
            
        else:
           #If it's a subfolder, recursively call rename files on that directory.
           print("Recursing: " + old_name)
           rename_files(old_name, 0)
    print("Done: File renaming for folder" + base_directory)

def channel_correction(base_directory):
    print ("In progress: channel correction " + base_directory)
    for path in os.listdir(base_directory):
    
        old_name = base_directory + '/' + path
        
        #convert file to jpg format
        if os.path.isfile(old_name):
            # checking number of dimensions of the image and delete images with less than 3 dimensions
            if(split_name(path)[1]).lower() in ['jpg', 'jpeg', 'png', 'jfif']:
                im = Image.open(old_name)
                shape_img = np.asarray(im).shape
                if len(shape_img) != 3:
                    new_im = im.convert("RGB")
                    os.remove(old_name)
                    new_im.save(old_name)
                if len(shape_img) == 3:
                    if shape_img[2] != 3:
                        print("Converted to RGB: " + old_name)
                        new_im = im.convert("RGB")
                        os.remove(old_name)
                        new_im.save(old_name)
                
            else:
                print("Deleting file " + old_name)
                os.remove(old_name)
        else:
           #If it's a subfolder, recursively call rename files on that directory.
           print("Recursing: " + old_name)
           channel_correction(old_name)
    print ("Done: channel correction " + base_directory)

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset divider")
    parser.add_argument("--data_path", required=True,
    help="Path to data")
    parser.add_argument("--test_data_path", required=False,
    help="Path to test data where to save")
    parser.add_argument("--train_ratio", required=True,
    help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test")
    parser.add_argument("--validation_ratio", required=False,
    help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test")
    parser.add_argument("--test_ratio", required=True,
    help="Train ratio - 0.7 means splitting data in 70 % train and 30 % test")

    return parser.parse_args()

           
if __name__ == '__main__':
    #dataset_path = '/Users/neerajmenon/Documents/Proj/gan/dim_correction'
    args = parse_args()
    dataset_path = args.data_path

    #converts all files to jpg format
    convert_to_jpg(dataset_path)

    #all images will be converted to RGB format (3 channels)
    channel_correction(dataset_path)

    #Covid files will have "COVID" in their name as prefix
    #Non-Covid files will have "Non-COVID" in their name as prefix
    rename_files(dataset_path, 0)

    #Splits the dataset into train and test with respective ratio
    splitfolders.ratio(dataset_path, output=dataset_path+"/data", seed=1337, ratio=(float(args.train_ratio), float(args.test_ratio))) # default values


# command 
# python preprocessing_helper.py --data_path "/Users/yadavpo/Documents/Proj/gan/dataset/CT" --train_ratio 0.7 --test_ratio 0.3
