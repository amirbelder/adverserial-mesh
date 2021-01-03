## ToDo
### Must
- Go through code, remove redundant
- Improve this README
- Check installations & requ.
- Fork, copy & pool request

### Optional
- Add eval to an arbitrary mesh - make it work -> norm model, 360 X 3 axis augmentation
- Eval Postprocess
- Support some meshes in one batch
- Generate other datasets
- Parallel run to generate walks

<img src='/doc/images/teaser_fig.png'>

# MeshWalker: Deep Mesh Understanding by Random Walks
Created by [Alon Lahav](mailto:alon.lahav2@gmail.com).
## SIGGRAPH Asia 2020
\[[Project Page](https://github.com/AlonLahav/MeshWalker)\]
\[[Video](to be added later)\]
\[[Paper](https://arxiv.org/abs/2006.05353)\]

This code implements the paper with some minor changes. 
In order to reproduce the paper results, please go to our [Project Page](https://github.com/AlonLahav/MeshWalker)\.

## Introduction
MeshWalker learns the shape directly from a given 3D mesh. 
The key idea is to represent the mesh by random walks along the surface, 
which "explore" the mesh’s geometry and topology. 
These walks form a list of vertices, which in some manner impose regularity on the mesh. 
They are fed into a Recurrent Neural Network (RNN) that "remembers" the history of the walk. 

## Installations
Use <a href="https://www.tensorflow.org/install/gpu"> this web page</a> to set up GPU support.

The following was created using <a href="https://www.tensorflow.org/install/pip?lang=python3">TF "Install TensorFlow with pip" page</a>.

1. Install Python related:
```
sudo apt update
sudo apt install python3-dev python3-pip python3-tk
sudo pip3 install -U virtualenv  # system-wide install
```

2. Clone, create a virtual environment and install packages
```
git clone https://github.com/tensorflow/graphics.git
cd graphics/tensorflow_graphics/projects/mesh_walker
virtualenv -p python3 ~/venv/mesh_walker_tfg
source ~/venv/mesh_walker_tfg/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
Use the virtual env & go to the directory:
```
source ~/venv/mesh_walker/bin/activate
cd graphics/tensorflow_graphics/projects/mesh_walker
```

To train model from scratch:
```
python train_eval.py
```
For the `dancer` dataset, you may expect good results after about 100k iterations 
(14 hours using NVIDIA-GTX1080ti).
To get the best results, wait for full convergence (~260k iterations, ~40 hours).

Check accuracy using `tensorboard`:
```
tensorboard --logdir /tmp/mesh_walker
```
Notice that `accuacy_test/per_walk` 
is related to the accuracy without averaging some walks.
`accuacy_test/full` is the results we get using the paper's eq.5.

Mesh results will be dumped occasionally under `/tmp/mesh_walker/output_dump`.


To use pre-trained model, downlowd it from `TBD`. Then run:
```
python train_eval.py --pretrained_model_path <path-to-keras-file>
``` 

## Reference
If you find our code or paper useful, please consider citing:
```
@article{lahav2020meshwalker,
  title={MeshWalker: Deep Mesh Understanding by Random Walks},
  author={Lahav, Alon and Tal, Ayellet},
  journal={arXiv preprint arXiv:2006.05353},
  year={2020}
}
```

<img src='/doc/images/2nd_fig.png'>
