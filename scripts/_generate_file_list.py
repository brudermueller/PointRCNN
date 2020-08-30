#!/usr/bin/env python
'''
    Script generating a text file with all files to be used for training and testing 
    from custom dataset.
'''
import os 

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/custom_data/")


def get_data_files(path, distance=8):
    """ Walk through given directory and corresponding subdirectories to generate a text file 
        with all training/testing instances in a list, like it has been done for the KITTI dataset.

    Args:
        path (string): path to main data directory containing all subdirectories from different 
                       recordings.
        distance (int, optional): Ignore recordings with objects being further away than the given 
                                  distance. Defaults to 8.

    Returns:
        [type]: [description]
    """
    folders_of_interest = [] 
    data_files = []                                    
    for root, dirs, files in os.walk(path):
        for folder in dirs: 
            dist = int(folder.split('_')[-1])
            if dist <= 8: 
                path = os.path.join(root, folder)
                folders_of_interest.append(path) 
    
    for path in folders_of_interest: 
        folder = path.split("/")[-1]
        data_files += [os.path.join(folder, f) for f in os.listdir(path) if f.endswith('.h5')]
    return data_files


if __name__=='__main__': 
    files = get_data_files(DATA_DIR, )
    with open(os.path.join(DATA_DIR, 'full_data.txt'), 'w') as f:
        for item in files:
            f.write("%s\n" % item)