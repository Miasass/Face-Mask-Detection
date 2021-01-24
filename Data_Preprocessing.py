# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import shutil
import xml.etree.ElementTree as ET
import cv2

def all_xml_to_df(path,name_of_csv_file):
    fnames = os.listdir(path)
    all_data=[]
    for fname in fnames:
        data=[]
        xml_data = open(path+'/'+fname, 'r').read()
        root = ET.XML(xml_data)  

        for i, child in enumerate(root):
            if i>=4:
                data.append([subchild.text for subchild in child if subchild.tag == 'name'])
                for subchild in child:
                    for subsubchild in subchild:
                        data[i-4].append(subsubchild.text)
    
        for i in range(len(data)):
            data[i].append(fname)
        
        all_data.append(data)
    
    cols = ['state_of_mask','xmin','ymin','xmax','ymax','file_name']
    for i,data in enumerate(all_data):
        if i==0:
            df = pd.DataFrame(data)
        else:
            temp_DF = pd.DataFrame(data)
            df = df.append(temp_DF,ignore_index=True)
            
    df.columns = cols
    df.to_csv(name_of_csv_file)
    
def create_new_images(path_to_csv_file,path_to_images,path_to_save_new_images,name_of_csv_file):
    df=pd.read_csv(path_to_csv_file,index_col=0)
    for i in range(len(df)):
        name_of_folder = df.loc[i][0]
        img = cv2.imread(filename=path_to_images+df.loc[i][-1].split('.')[0]+'.png')
        face = img[int(df.loc[i][2]):int(df.loc[i][4]),int(df.loc[i][1]):int(df.loc[i][3])]
        path=path_to_save_new_images+name_of_folder+'/'
        file_name = str(i)+'.png'
        cv2.imwrite(filename=path+file_name,img=face)
        data = [df.loc[i][0],file_name]
        if i==0:
            df2 = pd.DataFrame([data])
        else:
            df2 = df2.append(pd.DataFrame([data]),ignore_index=True)
            
    df2.columns = ['state_of_mask','file_name']
    df2.to_csv(name_of_csv_file)


def create_dirs(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_with_mask_dir = os.path.join(train_dir, 'with_mask')
    train_without_mask_dir = os.path.join(train_dir, 'without_mask')
    train_mask_weared_incorrect_dir = os.path.join(train_dir, 'mask_weared_incorrect')

    valid_with_mask_dir = os.path.join(valid_dir, 'with_mask')
    valid_without_mask_dir = os.path.join(valid_dir, 'without_mask')
    valid_mask_weared_incorrect_dir = os.path.join(valid_dir, 'mask_weared_incorrect')

    test_with_mask_dir = os.path.join(test_dir, 'with_mask')
    test_without_mask_dir = os.path.join(test_dir, 'without_mask')
    test_mask_weared_incorrect_dir = os.path.join(test_dir, 'mask_weared_incorrect')

    for directory in (train_dir, valid_dir, test_dir):
        if not os.path.exists(directory):
            os.mkdir(directory)

    dirs = [train_with_mask_dir, train_without_mask_dir, train_mask_weared_incorrect_dir,
            valid_with_mask_dir, valid_without_mask_dir, valid_mask_weared_incorrect_dir,
            test_with_mask_dir, test_without_mask_dir, test_mask_weared_incorrect_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            
    return dirs 
        

def size_of_sets(with_mask_path,without_mask_path,mask_weared_incorrect_path):
    with_mask_fnames = os.listdir(with_mask_path)
    without_mask_fnames = os.listdir(without_mask_path)
    mask_weared_incorrect_fnames = os.listdir(mask_weared_incorrect_path)
    
    size_with_mask = len(with_mask_fnames)
    size_without_mask = len(without_mask_fnames)
    size_mask_weared_incorrect = len(mask_weared_incorrect_fnames)

    train_size_with_mask = int(np.floor(0.8 * size_with_mask))
    valid_size_with_mask = int(np.floor(0.1 * size_with_mask))
    test_size_with_mask = size_with_mask - train_size_with_mask - valid_size_with_mask

    train_size_without_mask = int(np.floor(0.8 * size_without_mask))
    valid_size_without_mask = int(np.floor(0.1 * size_without_mask))
    test_size_without_mask = size_without_mask - train_size_without_mask - valid_size_without_mask

    train_size_mask_weared_incorrect = int(np.floor(0.8 * size_mask_weared_incorrect))
    valid_size_mask_weared_incorrect = int(np.floor(0.1 * size_mask_weared_incorrect))
    test_size_mask_weared_incorrect = size_mask_weared_incorrect - train_size_mask_weared_incorrect - valid_size_mask_weared_incorrect

    train_idx_with_mask = train_size_with_mask
    valid_idx_with_mask = train_size_with_mask + valid_size_with_mask
    test_idx_with_mask = train_size_with_mask + valid_size_with_mask + test_size_with_mask

    train_idx_without_mask = train_size_without_mask
    valid_idx_without_mask = train_size_without_mask + valid_size_without_mask
    test_idx_without_mask = train_size_without_mask + valid_size_without_mask + test_size_without_mask

    train_idx_mask_weared_incorrect = train_size_mask_weared_incorrect
    valid_idx_mask_weared_incorrect = train_size_mask_weared_incorrect + valid_size_mask_weared_incorrect
    test_idx_mask_weared_incorrect = train_size_mask_weared_incorrect + valid_size_mask_weared_incorrect + test_size_mask_weared_incorrect
    
    
    indexes = [train_idx_with_mask,train_idx_without_mask,train_idx_mask_weared_incorrect,
               valid_idx_with_mask,valid_idx_without_mask,valid_idx_mask_weared_incorrect,
               test_idx_with_mask,test_idx_without_mask,test_idx_mask_weared_incorrect]
    
    return indexes,with_mask_fnames,without_mask_fnames,mask_weared_incorrect_fnames


def copy_images_to_folders(with_mask_fnames,without_mask_fnames,mask_weared_incorrect_fnames,
                           train_idx_with_mask,train_idx_without_mask,train_idx_mask_weared_incorrect,
                           valid_idx_with_mask,valid_idx_without_mask,valid_idx_mask_weared_incorrect,
                           test_idx_with_mask,test_idx_without_mask,test_idx_mask_weared_incorrect,
                           train_with_mask_dir,train_without_mask_dir,train_mask_weared_incorrect_dir,
                           valid_with_mask_dir, valid_without_mask_dir, valid_mask_weared_incorrect_dir,
                           test_with_mask_dir,test_without_mask_dir,test_mask_weared_incorrect_dir):
        
        
        for i, fname in enumerate(with_mask_fnames):
            if i <= train_idx_with_mask:
                src = os.path.join(r'./Data/images_after_preprocessing/with_mask',fname)
                dst = os.path.join(train_with_mask_dir, fname)
                shutil.copyfile(src, dst)
            elif train_idx_with_mask < i <= valid_idx_with_mask:
                src = os.path.join(r'./Data/images_after_preprocessing/with_mask',fname)
                dst = os.path.join(valid_with_mask_dir, fname)
                shutil.copyfile(src, dst)
            elif valid_idx_with_mask < i < test_idx_with_mask:
                src = os.path.join(r'./Data/images_after_preprocessing/with_mask',fname)
                dst = os.path.join(test_with_mask_dir, fname)
                shutil.copyfile(src, dst)

        for i, fname in enumerate(without_mask_fnames):
            if i <= train_idx_without_mask:
                src = os.path.join(r'./Data/images_after_preprocessing/without_mask',fname)
                dst = os.path.join(train_without_mask_dir, fname)
                shutil.copyfile(src, dst)
            elif train_idx_without_mask < i <= valid_idx_without_mask:
                src = os.path.join(r'./Data/images_after_preprocessing/without_mask',fname)
                dst = os.path.join(valid_without_mask_dir, fname)
                shutil.copyfile(src, dst)
            elif valid_idx_without_mask < i < test_idx_without_mask:
                src = os.path.join(r'./Data/images_after_preprocessing/without_mask',fname)
                dst = os.path.join(test_without_mask_dir, fname)
                shutil.copyfile(src, dst) 

        for i, fname in enumerate(mask_weared_incorrect_fnames):
            if i <= train_idx_mask_weared_incorrect:
                src = os.path.join(r'./Data\images_after_preprocessing\mask_weared_incorrect',fname)
                dst = os.path.join(train_mask_weared_incorrect_dir, fname)
                shutil.copyfile(src, dst)
            elif train_idx_mask_weared_incorrect < i <= valid_idx_mask_weared_incorrect:
                src = os.path.join(r'./Data\images_after_preprocessing\mask_weared_incorrect',fname)
                dst = os.path.join(valid_mask_weared_incorrect_dir, fname)
                shutil.copyfile(src, dst)
            elif valid_idx_mask_weared_incorrect < i < test_idx_mask_weared_incorrect:
                src = os.path.join(r'./Data\images_after_preprocessing\mask_weared_incorrect',fname)
                dst = os.path.join(test_mask_weared_incorrect_dir, fname)
                shutil.copyfile(src, dst)

def main():
    all_xml_to_df('./Data/annotations','mask_data.csv')
    create_new_images('mask_data.csv','./Data/images/','./Data/images_after_preprocessing/','mask_data_v2.csv')
    dirs = create_dirs('./Data/done_images')
    indexes,with_mask_fnames,without_mask_fnames,mask_weared_incorrect_fnames = size_of_sets('./Data/images_after_preprocessing\with_mask','./Data/images_after_preprocessing\without_mask', './Data/images_after_preprocessing\mask_weared_incorrect')
    copy_images_to_folders(with_mask_fnames,without_mask_fnames,mask_weared_incorrect_fnames,
                           indexes[0],indexes[1],indexes[2],
                           indexes[3],indexes[4],indexes[5],
                           indexes[6],indexes[7],indexes[8],
                           dirs[0],dirs[1],dirs[2],
                           dirs[3],dirs[4],dirs[5],
                           dirs[6],dirs[7],dirs[8])
    
    return 0

main()
    
    
    
    
    
    
    