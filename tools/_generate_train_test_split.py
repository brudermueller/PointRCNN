#!/usr/bin/env python
'''
    Script generating grouping all data files from custom dataset into train/test/val sets. 
'''
import os 
from sklearn.model_selection import train_test_split  


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/custom_data/")



def gen_split_files(data_files): 
    """ Create a train/test/val split of the data and save it to txt files. 

    Args:
        data_files [array]: array containing all path names of h5 files to include in database
    """
    print('Full length of data: {}'.format(len(data_files)))
    # create 50/50 train test split on randomly shuffled data instances 
    data_train_temp, data_test = train_test_split(data_files, shuffle=True, train_size=0.5)
    # choosing again 50/50 train/val split ƒwithout reshuffling data
    data_train, data_val = train_test_split(data_train_temp, shuffle=False, train_size=0.5)
    
    dataset = {'train': data_train, 'val': data_val, 'test': data_test}

    # Save splits to separate txt files 
    for filename, data in dataset.items(): 

        print('Saving {}-split with {} instances.\n'.format(filename, len(data)))
        with open(os.path.join(DATA_DIR, '{}.txt'.format(filename)), 'w') as f:
            for item in data:
                f.write("%s\n" % item)


if __name__=='__main__': 
    all_files_filename = 'full_data.txt'
    data_files = [x.strip() for x in open(os.path.join(DATA_DIR, all_files_filename)).readlines()]
    gen_split_files(data_files)