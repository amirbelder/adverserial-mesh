"""
Shows a mesh, without the walking
"""

import glob, os, copy, sys

import numpy as np
import utils


def show_model(model):
    utils.visualize_model(model['vertices'], model['faces'])
    return

def load_model_from_npz(npz_path):
    if npz_path.find(':') != -1:
        npz_path = npz_path.split(':')[1]
    mesh_data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    return mesh_data

def main():
    print("in main")

    """npz_paths = ['datasets_processed/shrec11/16-04_a/train/T295_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T348_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T364_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T184_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T301_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T304_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T454_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T343_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T142_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T303_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T117_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T485_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T234_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T579_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T402_not_changed_500.npz',
                 'datasets_processed/shrec11/16-04_a/train/T529_not_changed_500.npz']"""
    #npz_paths = ['datasets_processed/shrec11_raw_2k/T295_simplified_to_2000.npz']
    #bird2
    npz_paths = ['datasets_processed/shrec11/16-04_a/train/T500_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T520_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T358_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T546_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T80_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T96_not_changed_500.npz']
    #bird1 #no legs and wings down
    #npz_paths = ['datasets_processed/shrec11/16-04_a/train/T159_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T118_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T76_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T66_not_changed_500.npz'	,'datasets_processed/shrec11/16-04_a/train/T79_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T60_not_changed_500.npz']

    #dog1 weird legs and no tail
    #npz_paths = ['datasets_processed/shrec11/16-04_a/train/T476_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T144_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T136_not_changed_500.npz']
    #npz_paths = ['datasets_processed/shrec11/16-04_a/train/T476_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T144_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T331_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T125_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T393_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T436_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T197_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T367_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T409_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T373_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T193_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T309_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T99_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T136_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T354_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T203_not_changed_500.npz']

    #dog2
    #npz_paths = ['datasets_processed/shrec11/16-04_a/train/T504_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T93_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T207_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T507_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T8_not_changed_500.npz']

    #man
    #npz_paths = ['datasets_processed/shrec11/16-04_a/train/T360_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T342_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T437_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T59_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T451_not_changed_500.npz']

    #2000
    #camel
    #'datasets_processed/shrec11_raw_2k/T398_simplified_to_2000.npz'
    #horse
    #'datasets_processed/shrec11_raw_2k/T295_simplified_to_2000.npz'

    # horse:
    # 'datasets_processed/shrec11/16-04_a/train/T348_not_changed_500.npz'
    # camel
    # 'datasets_processed/shrec11/16-04_a/train/T187_not_changed_500.npz'
    # 'datasets_processed/shrec11/16-04_a/train/T453_not_changed_500.npz'

    for npz_path in npz_paths:
        model = load_model_from_npz(npz_path)
        if model is not None:
            show_model(model)

    return 0


if __name__ == '__main__':
    print("amir")
    main()
"""""
    if len(sys.argv) > 1:
        npz_path = [sys.argv[1]]
    elif len(sys.argv) == 1:
        npz_paths = ['datasets_processed/shrec11/16-04_a/train/T295_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T348_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T364_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T184_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T301_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T304_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T454_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T343_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T142_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T303_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T117_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T485_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T234_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T579_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T402_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T529_not_changed_500.npz']
        #npz_path = 'datasets_processed/shrec11/16-04_a/train/T398_not_changed_500.npz'
        #horse:
        #'datasets_processed/shrec11/16-04_a/train/T348_not_changed_500.npz'
        # camel
        #'datasets_processed/shrec11/16-04_a/train/T187_not_changed_500.npz'
        #'datasets_processed/shrec11/16-04_a/train/T453_not_changed_500.npz'

        for npz_path in npz_paths:
            model = load_model_from_npz(npz_path)
            if model is not None:
                show_model(model)

    else:
        print("Error: Please provide .npz file path as an argument !")
        exit(0)

"""
