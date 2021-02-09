import time, glob, os, shutil, json

from tqdm import tqdm
import trimesh
from trimesh.transformations import euler_matrix
import numpy as np
from tqdm import tqdm
import pyvista as pv
import cv2

import utils
import walks
import dataset_prepare

'''
Generate video:
Windows:
1. open comman line
2. cd C:/Users/alon/mesh_walker/ant_video
3. C:/Users/alon\Downloads/ffmpeg-4.3.1-essentials_build/bin/ffmpeg -framerate 24 -i img_%04d.jpg ant_video.mp4
Ubuntu:
cd /home/alonlahav/mesh_walker/ant_video/
ffmpeg -framerate 24 -start_number 1 -i img_%04d.jpg -frames:v 250 ant_video_a.mp4
ffmpeg -framerate 200 -start_number 260 -i img_%04d.jpg ant_video_b.mp4
ffmpeg -framerate 200 -i img_%04d.jpg ant_video_all.mp4

ffmpeg -framerate 24 -i img_%04d.jpg ant_video_part-1.mp4
ffmpeg -framerate 160 -i img_%04d.jpg ant_video_part-2.mp4
'''

opacity = 0.4
walk_on_sphare = 0    # For debug
part = 1
show_frame_num = 0

human_seg_cmap = ['black', 'red', 'green', 'blue', 'orange', 'magenta', 'yellow', 'cyan', 'lightgreen']
colors_from_gur_paper = ['#FFB8B8', '#C5C5FF', '#FFD2FF', '#FFCD88', '#FF9966']

if 0:
  shape_id = 2
  path_output = os.path.expanduser('~') + '/runs/images/'
  HUMAN_MESH_FN = path_output + 'human_seg' + '/my_segmentation_res_' + str(shape_id) + '.npz'
elif 0:
  HUMAN_MESH_FN = os.path.expanduser('~') + '/mesh_walker/datasets_raw/sig17_seg_benchmark/meshes/test/shrec/2.off'
else:
  shape_id = 2
  path_output = os.path.expanduser('~') + '/mesh_walker/runs_pretrained/0001-09.08.2020..15.19__human_seg-6k_unit_bal_norm-4upload/'
  HUMAN_MESH_FN = path_output + '/seg_res__shape_' + str(shape_id) + '-b.npz'
ANT_MESH_FN = os.path.expanduser('~') + '/mesh_walker/datasets_raw/arbitrary_meshes/ant.obj'
#ANT_MESH_FN = os.path.expanduser('~') + '/mesh_walker/datasets_raw/arbitrary_meshes/uploads_files_737063_Ant_aligned.ply'


def get_rot_mat_using_up_and_los_vectors(up, los):
  # All vectors should be vertical ones

  # Make sure up vector is normalized
  up = up / np.linalg.norm(up)
  los = los / np.linalg.norm(los)

  # Change los to be ortogonal
  los = np.cross(up.T, np.cross(los.T, up.T)).T
  los = -los / np.linalg.norm(los)

  # Calc side vector
  side = np.cross(up.T, los.T).T
  side = side / np.linalg.norm(side)

  # Stack the rotation matrix
  rot_mat = np.hstack((side, los, -up)).T

  return rot_mat


def read_mesh_and_results_from_npz(fn):
  mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
  human_mesh = trimesh.Trimesh(mesh_data['vertices'], mesh_data['faces'], process=False)

  return human_mesh, mesh_data


def norm_meshes(mesh_hum, mesh_ant, ant_resize):
  if ANT_MESH_FN.endswith('ant.obj'):
    mesh_ant.vertices = mesh_ant.vertices[:, [0, 2, 1]]
    f = 0.06
  else:
    mesh_ant.vertices[:, 2] = -mesh_ant.vertices[:, 2]
    f = 0.02
  f = f * ant_resize
  mesh_ant.vertices = mesh_ant.vertices - [mesh_ant.vertices[:, 0].mean(), mesh_ant.vertices[:, 1].mean(), mesh_ant.vertices[:, 2].min()]
  mesh_ant.vertices *= f / np.max(mesh_ant.vertices)
  if mesh_hum is not None:
    mesh_hum.vertices = mesh_hum.vertices - [0, mesh_ant.vertices[:, 1].min(), 0]
    mesh_hum.vertices = mesh_hum.vertices[:, [0, 2, 1]]


def calc_ant_pos(mesh_hum, walk, step):
  p0 = mesh_hum.vertices[walk[int(step)]]
  p1 = mesh_hum.vertices[walk[int(step) + 1]]
  fr = step - int(step)
  ant_pos = p0 * (1 - fr) + p1 * fr
  up = mesh_hum.vertex_normals[walk[int(step) + 1]][:, np.newaxis]
  ant_los = (mesh_hum.vertices[walk[int(step)]] - mesh_hum.vertices[walk[int(step) + 1]])[:, np.newaxis]
  ant_rotation_mat = get_rot_mat_using_up_and_los_vectors(up, ant_los)

  return ant_pos, ant_rotation_mat


def create_image(mesh_hum, mesh_ant, ant_pos, walk2show, cpos, edge_colors_list, off_screen=True):
  if off_screen:
    window_size = [1920, 1080]
    p = pv.Plotter(off_screen=1, window_size=(int(window_size[0]), int(window_size[1])))
  else:
    p = pv.Plotter()

  # Human mesh
  faces = np.hstack([[3] + f.tolist() for f in mesh_hum.faces])
  surf = pv.PolyData(mesh_hum.vertices, faces)
  p.add_mesh(surf, show_edges=False, color=colors_from_gur_paper[0], opacity=opacity)

  # Edge colors according to the predictions:
  if edge_colors_list:
    for clr, edges in edge_colors_list.items():
      vertices2use = mesh_hum.vertices
      this_edges_ = [[2, e[0], e[1]] for e in edges]
      this_edges = np.hstack([edge for edge in this_edges_])
      walk_mesh = pv.PolyData(vertices2use, this_edges)
      line_width = 5 # 0.5
      p.add_mesh(walk_mesh, show_edges=True, edge_color=clr, line_width=line_width)

  # Walk
  if 0:#len(walk2show) > 2:
    walk_edges = [[2, walk2show[i], walk2show[i + 1]] for i in range(len(walk2show) - 1)]
    walk_edges = np.hstack([edge for edge in walk_edges])
    walk_mesh = pv.PolyData(mesh_hum.vertices, walk_edges)
    p.add_mesh(walk_mesh, show_edges=True, edge_color='blue', line_width=4)

  # Ant mesh
  faces = np.hstack([[3] + f.tolist() for f in mesh_ant.faces])
  surf = pv.PolyData(mesh_ant.vertices + ant_pos, faces)
  p.add_mesh(surf, show_edges=False, color=colors_from_gur_paper[4])

  p.camera_position = cpos
  p.set_background("#AAAAAA", top="White")

  if off_screen:
    rendered = p.screenshot()
    p.close()
  else:
    rendered = None
    cpos = p.show()
    utils.print_cpos(cpos)

  return rendered, cpos


def create_edge_colors_list(faces, vertex_predictions, human_seg_cmap):
  def _calc_edge_pred(p0, p1):
    p = (p0 + p1) / 2
    if np.max(p) < 0.1:
      return None
    else:
      return np.argmax(p)
  edge_colors_list = {}
  for face in faces:
    for f in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
      edge_pred = _calc_edge_pred(vertex_predictions[f[0]], vertex_predictions[f[1]])
      if edge_pred:
        color = human_seg_cmap[edge_pred]
        if color not in edge_colors_list.keys():
          edge_colors_list[color] = []
        edge_colors_list[color].append(f)
  return edge_colors_list

def walk2edge_colors_list(walk):
  color = 'blue'
  edge_colors_list = {color: []}
  for i in range(len(walk) - 2):
    edge_colors_list[color].append((walk[i], walk[i + 1]))
  edge_colors_list['red'] = [(walk[-2], walk[-1])]
  return edge_colors_list


def make_ant_movie():
  def _get_cpos(n_rendered, n_total):
    if part == 1:  # Neck
      cposes = [
        (-1, [[0.76 , -0.47 , 0.93] , [0.0 , -0.03 , 0.72] , [-0.2 , 0.12 , 0.97]]),
        (np.inf, [[0.86 , 0.61 , 0.83] , [0.0 , -0.03 , 0.72] , [-0.04 , -0.11 , 0.99]])
      ]
    elif part == 2:
      cposes = [
        (-1,     [[-0.35 , 0.42 , -0.52] , [0.17 , 0.1 , -0.39] , [-0.29 , -0.08 , 0.95]]),
        (200,    [[1.99 , 3.22 , 0.58] , [-0.03 , 0.01 , -0.02] , [-0.11 , -0.12 , 0.99]]),
        (np.inf, [[1.99 , 3.22 , 0.58] , [-0.03 , 0.01 , -0.02] , [-0.11 , -0.12 , 0.99]]),
      ]
    elif 0:     # For debug (walk on sphare)
      cposes = [
        (-1, [[-0.25 , -0.36 , -3.29] , [0.01 , -1.01 , 0.0] , [-0.9 , 0.41 , 0.15]]),
        (np.inf, [[-0.25 , -0.36 , -3.29] , [0.01 , -1.01 , 0.0] , [-0.9 , 0.41 , 0.15]]),
      ]
    else:       # hole body
      cposes = [
        (-1,     [[3.52 , -1.93 , 1.01] , [0.07 , 0.12 , -0.03] , [-0.22 , 0.11 , 0.97]]),
        (np.inf, [[3.52 , -1.93 , 1.01] , [0.07 , 0.12 , -0.03] , [-0.22 , 0.11 , 0.97]])
      ]
    for n in range(len(cposes)):
      if cposes[n][0] < n_rendered and cposes[n + 1][0] >= n_rendered:
        next_part = min(cposes[n + 1][0], n_total)
        f = (n_rendered - cposes[n][0]) / (next_part - cposes[n][0])
        cpos = np.array(cposes[n][1]) * (1 - f) + np.array(cposes[n + 1][1]) * f
        break
    return cpos

  off_screem = 1
  dump_path = os.path.expanduser('~') + '/mesh_walker/ant_video__part-' + str(part)
  if os.path.isdir(dump_path):
    shutil.rmtree(dump_path)
  os.makedirs(dump_path)

  # Load human & ant mesh
  if walk_on_sphare:
    mesh_hum = trimesh.primitives.Sphere()
    mesh_hum = trimesh.Trimesh(mesh_hum.vertices, mesh_hum.faces)
    predictions = None
    mesh_data = {'vertices': mesh_hum.vertices, 'faces': mesh_hum.faces, 'n_vertices': mesh_hum.vertices.shape[0]}
    dataset_prepare.prepare_edges_and_kdtree(mesh_data)
    walk, _ = walks.get_seq_random_walk_no_jumps(mesh_data, f0=0, seq_len=1000)
  elif HUMAN_MESH_FN.endswith('npz'):
    mesh_hum, mesh_hum_results = read_mesh_and_results_from_npz(HUMAN_MESH_FN)
    predictions = mesh_hum_results['predictions4vertex']      # [(10000-1) x 9]
    walk = mesh_hum_results['walk']                           # [10000]
  else:
    mesh_hum = trimesh.load_mesh(HUMAN_MESH_FN)

  # Calc prediction per step
  edge_colors_list = None
  if predictions is not None:
    pred_per_vertex = np.zeros((mesh_hum.vertices.shape[0], predictions.shape[1]))

  # Load ant mesh and normalize meshes for display
  mesh_ant = trimesh.load_mesh(ANT_MESH_FN)

  # for stem in walk
  if part == 1:
    n_step_start = 5615
    n_step_end = n_step_start + 50
    rot_stride = 0
    edges2show = 'walk'
    text = 'Random Walk'
    split_step_list = [50, 5]         # split_step_list[0] : when to have step=1 ; split_step_list[1] : begining step split
    ant_resize = 1
  else:
    n_step_start = 0
    n_step_end = n_step_start + 4000
    rot_stride = 360 / 1000
    edges2show = 'seg_res'
    text = 'Mesh Semantic Segmentation'
    split_step_list = [10, 10]
    ant_resize = 2

  norm_meshes(mesh_hum, mesh_ant, ant_resize)

  n_total = (n_step_end - n_step_start - split_step_list[0]) * 1 + split_step_list[0] * split_step_list[1]

  origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
  Rz = trimesh.transformations.rotation_matrix(rot_stride * np.pi / 180, zaxis)

  n_rendered = 0
  for i in tqdm(range(n_step_start, n_step_end)):#walk.size - 1):
    if predictions is not None:
      pred_per_vertex[walk[i]] += predictions[i]
    split_step_to = 1
    if i - n_step_start < split_step_list[0]:
      split_step_to = split_step_list[1]
    for frac_step in np.linspace(0, 1, split_step_to, endpoint=False):
      # Rotate mesh
      mesh_hum.vertices = trimesh.transformations.transform_points(mesh_hum.vertices, Rz)

      # Calc ant pos and angles
      ant_pos, ant_rotation_mat = calc_ant_pos(mesh_hum, walk, i + frac_step)

      # Calc camera position (range)

      # Rotate 2 meshes (camera angles are fixed)
      # Rotation angles change should be small (averaged over time)
      rotated_vertices = np.dot(mesh_ant.vertices, ant_rotation_mat)
      mesh_ant_rot = trimesh.Trimesh(rotated_vertices, mesh_ant.faces)

      # Calculate colors according to the prediction
      if predictions is not None and edges2show == 'seg_res':
        edge_colors_list = create_edge_colors_list(mesh_hum.faces, pred_per_vertex, human_seg_cmap)
      elif edges2show == 'walk' and i - n_step_start > 1:
        edge_colors_list = walk2edge_colors_list(walk[n_step_start:i + 1])

      # Create an image
      walk2show = walk[:i]
      cpos = _get_cpos(n_rendered, n_total)
      img, cpos = create_image(mesh_hum, mesh_ant_rot, ant_pos, walk2show, cpos, edge_colors_list, off_screen=off_screem)

      if not off_screem:
        exit(0)

      # Dump the image
      if img is not None:
        img = img.copy()
        if show_frame_num:
          cv2.putText(img, str(i), (img.shape[1] - 100, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        img = img[:, :, ::-1]
        cv2.imwrite(dump_path + '/img_' + str(n_rendered).zfill(4) + '.jpg', img)

      n_rendered += 1

  # Using ffmpeg (?) -> images 2 videos
  pass


def check_rotations():
  if 0:
    shp_a = trimesh.primitives.Cylinder()
    shp_b = trimesh.primitives.Box()
    vertices = np.vstack((shp_a.vertices + [0, 2, 0], shp_b.vertices - [0, 2, 0]))
    faces = np.vstack((shp_a.faces, shp_b.faces + shp_a.vertices.shape[0]))
    shp = trimesh.Trimesh(vertices, faces)
    cpos = [[14.03 , 14.03 , 14.03] , [0.0 , 0.0 , 0.0] , [0.0 , 0.0 , 1.0]]
  else:
    shp = trimesh.load_mesh(ANT_MESH_FN)
    norm_meshes(None, shp)
    shp.vertices = 10 * shp.vertices
    cpos = [[2.33 , 2.43 , 2.81] , [-0.19 , -0.09 , 0.28] , [0.0 , 0.0 , 1.0]]

  if 0:
    up =  np.array([[0.,   1.,  0.]]).T
    los = np.array([[0.,   0.,  0.1]]).T
  else:
    up =  np.array([[0.,   1.,  0.]]).T
    los = np.array([[0.1,  0.,  0.]]).T
  rot_mat = get_rot_mat_using_up_and_los_vectors(up, los)
  print(rot_mat)
  #shp.vertices = np.dot(shp.vertices, rot_mat)

  p = pv.Plotter()
  p.camera_position = cpos
  faces = np.hstack([[3] + f.tolist() for f in shp.faces])
  surf = pv.PolyData(shp.vertices, faces)
  p.add_mesh(surf, show_edges=True)
  p.show_axes()
  p.show_grid()
  cpos = p.show()
  utils.print_cpos(cpos)


if __name__ == '__main__':
  np.random.seed(0)
  if 0:
    check_rotations()
  else:
    make_ant_movie()
