"""General purpose visualizer for semantic segmentation results on various datasets super-fueled by open3D.
"""
import os
import open3d
import torch
import tensorflow as tf
import numpy as np
from termcolor import colored
# from base.base_dataset import BaseDataSet

pos_neg_map = np.asarray(
  [
    [200, 200, 200],
    [0, 255, 0],
    [255, 0, 0]], type=np.int32)

color_map = np.asarray([[255, 255, 255],  # unlabeled
     [174, 199, 232],  # wall
     [152, 223, 138],  # floor
     [31, 119, 180],  # cabinet
     [255, 187, 120],  # bed
     [188, 189, 34],  # chair
     [140, 86, 75],  # sofa
     [255, 152, 150],  # table
     [214, 39, 40],  # door
     [197, 176, 213],  # window
     [148, 103, 189],  # bookshelf
     [196, 156, 148],  # picture
     [23, 190, 207],  # counter
     [247, 182, 210],  # desk
     [219, 219, 141],  # curtain
     [255, 127, 14],  # refrigerator
     [158, 218, 229],  # shower curtain
     [44, 160, 44],  # toilet
     [112, 128, 144],  # sink
     [227, 119, 194],  # bathtub
     [82, 84, 163]], type=np.int32)  # otherfurn


class SemSegVisualizer:
  ''' ---  https://github.com/VisualComputingInstitute/dcm-net --- '''
  """Visualize meshes from various datasets with open3D. Key Events show RGB, ground truth, prediction or differences."""

  def __init__(self, save_dir: str = ""):
    """Initialize Semantic Segmentation Visualizer which shows meshes with optional prediction and ground truth

    Arguments:
        save_dir {str}         -- Directory in which .ply files should be saved
    """


    if not os.path.exists(os.path.dirname(save_dir)):
      os.makedirs(os.path.dirname(save_dir))

    assert os.path.isdir(save_dir) or save_dir == ""
    self._save_dir = save_dir

  def visualize_result(self, mesh_name, prediction=None, gt=None):
    mesh = open3d.read_triangular_mesh(mesh_name)
    mesh.compute_vertex_normals()

    vis = open3d.VisualizerWithKeyCallback()
    vis.create_window(width=1600, height=1200)

    # PREPARE RGB COLOR SWITCH
    rgb_colors = open3d.Vector3dVector(np.asarray(mesh.vertex_colors))

    def colorize_rgb(visu):
      mesh.vertex_colors = rgb_colors
      visu.update_geometry()
      visu.update_renderer()

    vis.register_key_callback(ord('H'), colorize_rgb)

    if type(prediction) == np.ndarray:
      # PREPARE PREDICTION COLOR SWITCH
      pred_colors = open3d.Vector3dVector(color_map[prediction] / 255.)

      def colorize_pred(visu):
        mesh.vertex_colors = pred_colors
        visu.update_geometry()
        visu.update_renderer()

      vis.register_key_callback(ord('J'), colorize_pred)

    if type(gt) ==np.ndarray:
      # PREPARE GROUND TRUTH COLOR SWITCH
      gt_colors = open3d.Vector3dVector(color_map[gt.long()] / 255.)

      def colorize_gt(visu):
        mesh.vertex_colors = gt_colors
        visu.update_geometry()
        visu.update_renderer()

      vis.register_key_callback(ord('K'), colorize_gt)

    if type(gt) == np.ndarray and type(prediction) == np.ndarray:
      # PREAPRE DIFFERENCE COLOR SWITCH
      pos = (prediction == gt)
      neg = ((prediction != gt) & (gt != 0)) * 2

      differences = pos + neg
      diff_colors = open3d.Vector3dVector(pos_neg_map[differences.astype(np.int32)] / 255.)

      def colorize_diff(visu):
        mesh.vertex_colors = diff_colors
        visu.update_geometry()
        visu.update_renderer()

      vis.register_key_callback(ord('F'), colorize_diff)

    def save_room(visu):
      mesh.vertex_colors = rgb_colors
      open3d.io.write_triangle_mesh(
        f"{self._save_dir}/SemSegVisualizer_rgb.ply", mesh)

      if type(prediction) == torch.Tensor:
        mesh.vertex_colors = pred_colors
        open3d.io.write_triangle_mesh(
          f"{self._save_dir}/SemSegVisualizer_pred.ply", mesh)

      if type(gt) == torch.Tensor:
        mesh.vertex_colors = gt_colors
        open3d.io.write_triangle_mesh(
          f"{self._save_dir}/SemSegVisualizer_gt.ply", mesh)

      if type(gt) == torch.Tensor and type(prediction) == torch.Tensor:
        mesh.vertex_colors = diff_colors
        open3d.io.write_triangle_mesh(
          f"{self._save_dir}/SemSegVisualizer_diff.ply", mesh)
      print(colored(
        f"PLY meshes successfully stored in {os.path.abspath(self._save_dir)}", 'green'))

    vis.register_key_callback(ord('D'), save_room)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()



if __name__ == '__main__':
  import glob
  run_name = '0113-10.11.2020..15.51__scannet_v2_attention'
  gts = glob.glob('/home/ran/mesh_walker/runs_aug_360_must/{}/eval_scannet/gt/*.txt'.format(run_name))
  preds = glob.glob('/home/ran/mesh_walker/runs_aug_360_must/{}/eval_scannet/pred/*.txt'.format(run_name))
  # TODO: i need the PLY for xyz themselves
  scenes = glob.glob('/home/ran/Databases/scannet_v2/scans/scene0004_00/*clean_2.ply')