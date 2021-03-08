from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from models.layers import mesh_prepare
import os
import numpy as np

def add_adverserial_examples(dataset):
    #dataset.dataset.paths = []
    return

def run_test(epoch=-1, vertices = None, faces= None, label= None, attack = False):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        if i==4 and attack == True:
            for i in range(4):
                attacked_data = mesh_prepare.rebuild_mesh(vertices, faces)
                #data['label'][0] = label
                data['edge_features'][-i-1] = attacked_data.features# data['mesh'][0].features
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


def extract_data_of_attacked_meshes(path_to_walker_meshes):
    paths = os.listdir(path_to_walker_meshes)
    paths_to_meshes = [path for path in paths if path.__contains__('_attacked')]

    for mesh_path in paths_to_meshes:
      orig_mesh_data = np.load(path_to_walker_meshes + mesh_path, encoding='latin1', allow_pickle=True)
      attacked_mesh_data = {k: v for k, v in orig_mesh_data.items()}
      vertices, faces, label = attacked_mesh_data['vertices'], attacked_mesh_data['faces'], attacked_mesh_data['label']
      run_test(vertices=vertices, faces=faces, label=label, attack=True)

if __name__ == '__main__':
    extract_data_of_attacked_meshes(path_to_walker_meshes = 'datasets_processed/shrec11/')
