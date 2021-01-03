# triplet loss - supervised
#best shrec11 was:
#MODEL_PATH="../../mesh_walker/runs_aug_360_must/0048-30.12.2020..17.19__shrec11_16-04_a"
#more than 99% accuracy
#python evaluate_clustering.py /home/amir//MeshWalker_UnsupervisedLearning-main/datasets_processed/shrec11/ $MODEL_PATH "shrec11" "accuracy"


MODEL_PATH="../../mesh_walker/runs_aug_360_must/0059-01.01.2021..23.14__modelnet"
python evaluate_clustering.py /home/amir/Desktop/modelnet40_1k2k4k/ $MODEL_PATH "modelnet40" "accuracy"


#"/home/amir/mesh_walker/runs_aug_360_must/0031-24.12.2020..16.03__shrec11_16-04_a/"
#"/home/amir/Desktop/MeshWalker_UnsupervisedLearning-main/"
#"/home/galye/mesh_walker/runs_aug_360_must/0161-18.12.2020..16.36__shrec11_16-4_A/"
# regular shrec11
#MODEL_PATH="/home/galye/mesh_walker/runs_aug_360_must/0085-26.11.2020..06.49__shrec11_16-4_A_seq_len_200/"
# regular modelnet40
#MODEL_PATH="/home/galye/mesh_walker/runs_aug_360_must/0139-11.12.2020..16.45__modelnet/"
#python evaluate_clustering.py /home/galye/mesh_walker/datasets_processed/shrec16/ latest shrec11 accuracy
#python evaluate_clustering.py /home/galye/mesh_walker/datasets_processed/shrec16/ $MODEL_PATH "shrec11" "accuracy"
#python evaluate_clustering.py /home/galye/mesh_walker/datasets_processed/modelnet40_1k2k4k/ $MODEL_PATH "modelnet40" "accuracy"
#python evaluate_clustering.py /home/galye/mesh_walker/datasets_processed/modelnet40_1k2k4k/ $MODEL_PATH "modelnet40" "tsne"
#python evaluate_clustering.py /home/galye/mesh_walker/datasets_processed/modelnet40/ $MODEL_PATH "modelnet40" "accuracy"

#python evaluate_clustering.py /home/galye/mesh_walker/datasets_processed/shrec16/ $MODEL_PATH "modelnet40" "tsne"


