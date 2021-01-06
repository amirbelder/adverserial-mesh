import numpy as np
import dataset
import pandas as pd

"""
Insert into csv the names of the different shrec11 labels,
and for each shows the names of the models that are relevant.  
"""

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]


def genertate_csv(filenames_, name: str):
  filenames = []
  files_labels = []
  data = {shrec11_labels[i]: list() for i in range(len(shrec11_labels))}
  for fn in filenames_:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    data[shrec11_labels[mesh_data['label']]].append(fn)
    #label_name = shrec11_labels[mesh_data['label']]
    #filenames.append(fn)
    #files_labels.append(label_name)
  #data = {'File name': filenames, 'Label': files_labels}
  csv_data = pd.DataFrame.from_dict(data, orient="index")
  csv_data.to_csv("files and labels " + name + ".csv")
  return


def tf_mesh_dataset(params, pathname_expansion, min_max_faces2use=[0, np.inf], data_augmentation={}, name: str = "train"):
  params_idx = dataset.setup_dataset_params(params, data_augmentation)
  number_of_features = dataset.dataset_params_list[params_idx].number_of_features
  params.net_input_dim = number_of_features

  filenames = dataset.get_file_names(pathname_expansion, min_max_faces2use)
  genertate_csv(filenames, name)

  return

