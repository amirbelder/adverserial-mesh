import glob, os, shutil, sys, json
from pathlib import Path

import trimesh
import open3d
from easydict import EasyDict
import numpy as np
from tqdm import tqdm

import pywavefront as pwf

# import utils

# Labels for all datasets
# -----------------------
sigg17_part_labels = ['---', 'head', 'hand', 'lower-arm', 'upper-arm', 'body', 'upper-lag', 'lower-leg', 'foot']
sigg17_shape2label = {v: k for k, v in enumerate(sigg17_part_labels)}

model_net_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser',
  'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp',
  'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood',
  'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl',
  'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]
model_net_shape2label = {v: k for k, v in enumerate(model_net_labels)}

cubes_labels = [
  'apple',  'bat',      'bell',     'brick',      'camel',
  'car',    'carriage', 'chopper',  'elephant',   'fork',
  'guitar', 'hammer',   'heart',    'horseshoe',  'key',
  'lmfish', 'octopus',  'shoe',     'spoon',      'tree',
  'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}

shrec17_labels = ['cat', 'centaur', 'david', 'dog', 'gorilla', 'horse', 'kid', 'michael', 'victoria', 'wolf']
shrec17_shape2label = {v: k for k, v in enumerate(shrec17_labels)}

coseg_labels = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}

s3dis_labels = ['ceiling', 'floor', 'wall', 'beam', 'column',
                'window', 'door', 'table', 'chair', 'sofa',
                'bookcase', 'board', 'clutter', '<UNK>']
s3dis_shape2label = {v: k for k, v in enumerate(s3dis_labels)}
s3dis_shape2label['<UNK>'] = -1

class WavefrontOBJ:
  def __init__(self, default_mtl='default_mtl'):
    self.path = None  # path of loaded object
    self.mtllibs = []  # .mtl files references via mtllib
    self.mtls = [default_mtl]  # materials referenced
    self.mtlid = []  # indices into self.mtls for each polygon
    self.vertices = []  # vertices as an Nx3 or Nx6 array (per vtx colors)
    self.normals = []  # normals
    self.texcoords = []  # texture coordinates
    self.polygons = []  # M*Nv*3 array, Nv=# of vertices, stored as vid,tid,nid (-1 for N/A)


def load_obj(filename: str, default_mtl='default_mtl', triangulate=False) -> WavefrontOBJ:
  """Reads a .obj file from disk and returns a WavefrontOBJ instance

  Handles only very rudimentary reading and contains no error handling!

  Does not handle:
      - relative indexing
      - subobjects or groups
      - lines, splines, beziers, etc.
  """

  # parses a vertex record as either vid, vid/tid, vid//nid or vid/tid/nid
  # and returns a 3-tuple where unparsed values are replaced with -1
  def parse_vertex(vstr):
    vals = vstr.split('/')
    vid = int(vals[0]) - 1
    tid = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else -1
    nid = int(vals[2]) - 1 if len(vals) > 2 else -1
    return (vid, tid, nid)

  with open(filename, 'r') as objf:
    obj = WavefrontOBJ(default_mtl=default_mtl)
    obj.path = filename
    cur_mat = obj.mtls.index(default_mtl)
    for line in objf:
      toks = line.split()
      if not toks:
        continue
      if toks[0] == 'v':
        obj.vertices.append([float(v) for v in toks[1:]])
      elif toks[0] == 'vn':
        obj.normals.append([float(v) for v in toks[1:]])
      elif toks[0] == 'vt':
        obj.texcoords.append([float(v) for v in toks[1:]])
      elif toks[0] == 'f':
        poly = [parse_vertex(vstr) for vstr in toks[1:]]
        if triangulate:
          for i in range(2, len(poly)):
            obj.mtlid.append(cur_mat)
            obj.polygons.append((poly[0], poly[i - 1], poly[i]))
        else:
          obj.mtlid.append(cur_mat)
          obj.polygons.append(poly)
      elif toks[0] == 'mtllib':
        obj.mtllibs.append(toks[1])
      elif toks[0] == 'usemtl':
        if toks[1] not in obj.mtls:
          obj.mtls.append(toks[1])
        cur_mat = obj.mtls.index(toks[1])
    return obj


def mesh_up_sampling(mesh_orig, target_n_faces):
  faces = np.asarray(mesh_orig.triangles)
  vertices = np.asarray(mesh_orig.vertices)
  #jitter_amp = (vertices.max(axis=0) - vertices.min(axis=0)).min() / 100
  new_faces = []
  new_vertices = []
  while faces.shape[0] < target_n_faces:
    for i_face, face in enumerate(faces):
      face_vertices = vertices[face, :]
      #jitter = 0#np.random.normal(size=3) * jitter_amp
      new_vertex = np.mean(face_vertices, axis=0) # + jitter
      new_v_idx = vertices.shape[0] + len(new_vertices)
      new_vertices.append(new_vertex)
      new_faces.append([face[0], face[1], new_v_idx])
      new_faces.append([face[1], face[2], new_v_idx])
      new_faces.append([face[2], face[0], new_v_idx])
    vertices = np.vstack((vertices, new_vertices))
    faces    = np.array(new_faces)
    new_vertices = []
    new_faces = []

  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(vertices)
  mesh.triangles = open3d.utility.Vector3iVector(faces)

  if 0:
    utils.visualize_model(vertices, faces,line_width=1, opacity=1.0)

  if 0:
    t_mesh_ = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    print(t_mesh_.is_winding_consistent)

  return mesh


def calc_vertex_labels_from_face_labels(mesh, face_labels):
  vertices = mesh['vertices']
  faces = mesh['faces']
  all_vetrex_labels = [[] for _ in range(vertices.shape[0])]
  vertex_labels = -np.ones((vertices.shape[0],), dtype=np.int)
  n_classes = int(np.max(face_labels))
  assert np.min(face_labels) == 1 # min label is 1, for compatibility to human_seg labels representation
  v_labels_fuzzy = -np.ones((vertices.shape[0], n_classes))
  for i in range(faces.shape[0]):
    label = face_labels[i]
    for f in faces[i]:
      all_vetrex_labels[f].append(label)
  for i in range(vertices.shape[0]):
    counts = np.bincount(all_vetrex_labels[i])
    vertex_labels[i] = np.argmax(counts)
    v_labels_fuzzy[i] = np.zeros((1, n_classes))
    for j in all_vetrex_labels[i]:
      v_labels_fuzzy[i, int(j) - 1] += 1 / len(all_vetrex_labels[i])
  return vertex_labels, v_labels_fuzzy


def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])]
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']])
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  if n_nbrs < 2:
    n_nbrs=2
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    try:
      i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    except:
      print('catched a problem - debug')
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])


def add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' or field == 'edges':
        prepare_edges_and_kdtree(m)
      # if field == 'edges':
      #   t_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
      #   m[field] = t_mesh.edges
      if field == 'angles':
        t_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
        m['adj_edges'] = t_mesh.face_adjacency_edges
        m['trimesh_vertices'] = t_mesh.vertices
        m[field] = t_mesh.face_adjacency_angles
      if field == 'face_normals':
        #TODO: fill this
        t_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
        m[field] = t_mesh.face_normals
      if field == 'adj_faces':
        adj_faces = get_faces_adjacent_faces(vertices=mesh_data.vertices, faces=mesh_data.faces)
        m[field] = adj_faces
      if field == 'tri_centers':
        t_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
        m[field] = t_mesh.triangles_center
      if field == 'vertex_normals':
        t_mesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces, process=False)
        m[field] = t_mesh.vertex_normals

  if dump_model:
    np.savez(out_fn, **m)

  return m


def get_sig17_seg_bm_labels(mesh, file, seg_path):
  # Finding the best match file name .. :
  in_to_check = file.replace('obj', 'txt')
  in_to_check = in_to_check.replace('off', 'txt')
  in_to_check = in_to_check.replace('_fix_orientation', '')
  if in_to_check.find('MIT_animation') != -1 and in_to_check.split('/')[-1].startswith('mesh_'):
    in_to_check = '/'.join(in_to_check.split('/')[:-2])
    in_to_check = in_to_check.replace('MIT_animation/meshes_', 'mit/mit_')
    in_to_check += '.txt'
  elif in_to_check.find('/scape/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/scape.txt'
  elif in_to_check.find('/faust/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/faust.txt'

  seg_full_fn = []
  for fn in Path(seg_path).rglob('*.txt'):
    tmp = str(fn)
    tmp = tmp.replace('/segs/', '/meshes/')
    tmp = tmp.replace('_full', '')
    tmp = tmp.replace('shrec_', '')
    tmp = tmp.replace('_corrected', '')
    if tmp == in_to_check:
      seg_full_fn.append(str(fn))
  if len(seg_full_fn) == 1:
    seg_full_fn = seg_full_fn[0]
  else:
    print('\nin_to_check', in_to_check)
    print('tmp', tmp)
    raise Exception('!!')
  face_labels = np.loadtxt(seg_full_fn)

  return face_labels


def get_labels(dataset_name, mesh, file, fn2labels_map=None):
  if dataset_name == 'faust':
    face_labels = np.load('faust_labels/faust_part_segmentation.npy').astype(np.int)
    vertex_labels, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh, face_labels)
    model_label = np.zeros((0,))
    return model_label, vertex_labels
  elif dataset_name.startswith('coseg') or dataset_name == 'human_seg_from_meshcnn':
    labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2] + '.eseg'
    e_labels = np.loadtxt(labels_fn)
    v_labels = [[] for _ in range(mesh['vertices'].shape[0])]
    faces = mesh['faces']

    edge2key = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
      faces_edges = []
      for i in range(3):
        cur_edge = (face[i], face[(i + 1) % 3])
        faces_edges.append(cur_edge)
      for idx, edge in enumerate(faces_edges):
        edge = tuple(sorted(list(edge)))
        faces_edges[idx] = edge
        if edge not in edge2key:
          edge2key[edge] = edges_count
          edges.append(list(edge))
          v_labels[edge[0]].append(e_labels[edges_count])
          v_labels[edge[1]].append(e_labels[edges_count])
          edges_count += 1

    vertex_labels = []
    for l in v_labels:
      l2add = np.argmax(np.bincount(l))
      vertex_labels.append(l2add)
    vertex_labels = np.array(vertex_labels)
    model_label = np.zeros((0,))
    return model_label, vertex_labels
  else:
    tmp = file.split('/')[-1]
    model_name = '_'.join(tmp.split('_')[:-1])
    if dataset_name.lower().startswith('modelnet'):
      model_label = model_net_shape2label[model_name]
    elif dataset_name.lower().startswith('cubes'):
      model_label = cubes_shape2label[model_name]
    elif dataset_name.lower().startswith('shrec11'):
      model_name = file.split('/')[-3]
      if fn2labels_map is None:
        model_label = shrec11_shape2label[model_name]
      else:
        file_index = int(file.split('.')[-2].split('T')[-1])
        model_label = fn2labels_map[file_index]
    elif dataset_name.lower().startswith('shrec17'):
      model_name = file.split('/')[-1].split('.')[0]
      if fn2labels_map is None:
        for k in shrec17_shape2label.keys():
          if k in model_name:
            model_label = shrec17_shape2label[k]
      else:
        file_index = int(file.split('.')[-2].split('T')[-1])
        model_label = fn2labels_map[file_index]
    else:
      raise Exception('Cannot find labels for the dataset')
    vertex_labels = np.zeros((0,))
    return model_label, vertex_labels


def fix_labels_by_dist(vertices, orig_vertices, labels_orig):
  labels = -np.ones((vertices.shape[0], ))

  for i, vertex in enumerate(vertices):
    d = np.linalg.norm(vertex - orig_vertices, axis=1)
    orig_idx = np.argmin(d)
    labels[i] = labels_orig[orig_idx]

  return labels


def get_faces_belong_to_vertices(vertices, faces):
  faces_belong = []
  for face in faces:
    used = np.any([v in vertices for v in face])
    if used:
      faces_belong.append(face)
  return np.array(faces_belong)


def get_faces_adjacent_faces(vertices, faces):
  tm_obj = trimesh.Trimesh(vertices, faces, remesh=False)
  adj_pairs = tm_obj.face_adjacency
  adj_arr = [[] for i in range(len(faces))]
  adj_mat = -1 * np.ones((len(faces), 3))
  for p in adj_pairs:
    adj_arr[p[0]].append(p[1])
    adj_arr[p[1]].append(p[0])
  for k, aa in enumerate(adj_arr):
    adj_mat[k, :len(aa)] = aa
  return adj_mat.astype(np.int16)


def remesh(mesh_orig, target_n_faces, add_labels=False, labels_orig=None):
  labels = labels_orig
  str_to_add = ''
  if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
    mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
    str_to_add += '_simplified_to_' + str(target_n_faces)
  else:
    mesh = mesh_orig
    while target_n_faces > np.asarray(mesh.triangles).shape[0]:
      mesh = mesh.subdivide_loop()
    str_to_add += '_upsampled_'
    mesh = mesh.simplify_quadric_decimation(target_n_faces)
    str_to_add += '_simplified_to_' + str(len(mesh.triangles))
    # if not len(mesh.triangles) == target_n_faces:
    #   print('debug')
    # str_to_add = '_not_changed_' + str(np.asarray(mesh_orig.triangles).shape[0])

  mesh = mesh.remove_unreferenced_vertices()

  if add_labels and labels_orig.size:
    labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)

  return mesh, labels, str_to_add


def load_meshes(model_fns):
  f_names = glob.glob(model_fns)
  joint_mesh_vertices = []
  joint_mesh_faces = []
  for fn in f_names:
    mesh_ = trimesh.load_mesh(fn)
    vertex_offset = len(joint_mesh_vertices)
    joint_mesh_vertices += mesh_.vertices.tolist()
    faces = mesh_.faces + vertex_offset
    joint_mesh_faces += faces.tolist()

  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(joint_mesh_vertices)
  mesh.triangles = open3d.utility.Vector3iVector(joint_mesh_faces)

  return mesh


def load_obj_mesh(model_fn):
  room_obj = load_obj(model_fn)
  faces_indices = np.asarray([(x[0][0], x[1][0], x[2][0]) for x in room_obj.polygons])
  vertices = np.asarray(room_obj.vertices)
  mesh_per_room = {}
  for i, x in enumerate(room_obj.mtlid):
    face_label = room_obj.mtls[x].split('_')[0]
    if face_label == '<UNK>':
      continue
    face_obj_instance = room_obj.mtls[x].split('_')[1]
    room_name = '_'.join(room_obj.mtls[x].split('_')[2:4])
    full_name = model_fn.split('/')[-3] + '_' + room_name
    mesh_per_room.setdefault(full_name, {'faces': []})['faces'].append(faces_indices[i])
    mesh_per_room[full_name].setdefault('face_label', []).append(s3dis_shape2label[face_label])
    mesh_per_room[full_name].setdefault('label_instance', []).append(face_obj_instance)

  # TODO: add unique vertices for each room, sort vertices & faces & vertex normals
  for k,v in mesh_per_room.items():
    all_verts_in_faces = list(np.unique(np.concatenate(v['faces'])[:]))
    room_verts = vertices[all_verts_in_faces]
    vertex_labels = np.zeros(len(all_verts_in_faces))
    new_faces = [[all_verts_in_faces.index(x) for x in y] for y in v['faces']]
    for f, l in zip(new_faces, v['face_label']):
      for ind in f:
        vertex_labels[ind] = l
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(room_verts)
    mesh.triangles = open3d.utility.Vector3iVector(new_faces)
    mesh_per_room[k]['mesh'] = mesh
    mesh_per_room[k]['label'] = vertex_labels
    assert mesh_per_room[k]['label'].shape[0] == np.asarray(mesh_per_room[k]['mesh'].vertices).shape[0]
  return mesh_per_room





  face_labels = [room_obj.mtls[x].split('_')[0] for x in room_obj.mtlid]
  # Ignore Unknown label vertices/faces
  unk_indices = [i for i,k in enumerate(face_labels) if k == '<UNK>']
  faces_indices = faces_indices[[x for x in range(faces_indices.shape[0]) if x not in unk_indices]]
  face_labels = [face_labels[z] for z in [x for x in range(len(face_labels)) if x not in unk_indices]]

  vertex_labels = -1 * np.ones(vertices.shape[0]).astype(np.int)
  unk_v_indices = []
  for i,l in zip(faces_indices, face_labels):
    if l in s3dis_shape2label.keys():
      vertex_labels[i] = s3dis_shape2label[l]
    else:
      unk_v_indices += list(i)
  # vertices = vertices[[x for x in range(vertices.shape[0]) if vertex_labels[x] != -1]]
  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(vertices)
  mesh.triangles = open3d.utility.Vector3iVector(faces_indices)
  return mesh, vertex_labels



def load_mesh(model_fn, classification=True):
  mesh_ = trimesh.load_mesh(model_fn, process=False)
  mesh_.remove_duplicate_faces()

  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
  mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)
  return mesh


# def create_tmp_dataset(model_fn, p_out, n_target_faces):
#   fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
#                    'label', 'labels', 'dataset_name']
#   if not os.path.isdir(p_out):
#     os.makedirs(p_out)
#   mesh_orig = load_mesh(model_fn)
#   mesh, labels, str_to_add = remesh(mesh_orig, n_target_faces)
#   labels = np.zeros((np.asarray(mesh.vertices).shape[0],), dtype=np.int16)
#   mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': 0, 'labels': labels})
#   out_fn = p_out + '/tmp'
#   add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, 'tmp')


def prepare_directory_segmentation(dataset_name, pathname_expansion, p_out, add_labels, fn_prefix, n_target_faces, classification=False):
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  for file in tqdm(filenames):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = mesh_orig = load_mesh(file, classification=classification)
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    label, labels_orig = get_labels(dataset_name, mesh_data, file)
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      out_fc_full = out_fn + str_to_add
      if os.path.isfile(out_fc_full + '.npz'):
        continue
      add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
      if 0:
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                              cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])
      if 0:
        utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])#, cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])


def fix_mesh_human_body_from_meshcnn(mesh, model_name, verbose=False):
  vertices = np.asarray(mesh.vertices)
  flip_orientation_fn = ['test/shrec__7.obj', 'test/shrec__8.obj', 'test/shrec__9.obj', 'test/shrec__1.obj',
                         'test/shrec__11.obj', 'test/shrec__12.obj']
  if np.any([model_name.endswith(to_check) for to_check in flip_orientation_fn]):
    if verbose:
      print('\n\nOrientation changed\n\n')
    vertices = vertices[:, [0, 2, 1]]
  if model_name.find('/scape/') != -1:
    if verbose:
      print('\n\nOrientation changed 2\n\n')
    vertices = vertices[:, [1, 0, 2]]
  if model_name.endswith('test/shrec__12.obj'):
    if verbose:
      print('\n\nScaling factor 10\n\n')
    vertices = vertices / 10
  if model_name.find('/adobe') != -1:
    if verbose:
      print('\n\nScaling factor 100\n\n')
    vertices = vertices / 100

  # Fix so model minimum hieght will be 0 (person will be on the floor). Up is dim 1 (2nd)
  vertices[:, 1] -= vertices[:, 1].min()
  mesh.vertices = open3d.utility.Vector3dVector(vertices)

  return mesh


def prepare_directory_from_scratch(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                   size_limit=np.inf, fn_prefix='', verbose=True, classification=True):
  fields_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'angles', 'face_normals', 'adj_faces', 'tri_centers',
                   'vertex_normals']

  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.splitext(os.path.basename(file))[0]
    if 'S3DIS' in file:
      # TODO: split each area to rooms, so we each room will be a sample
      # rooms, vertices, faces = load_s3dis_area(file)
      mesh_per_room = load_obj_mesh(file)
      for k, v in mesh_per_room.items():
        # if os.path.exists(p_out + '/' + k + '.npz'):
        #   continue
        mesh = v['mesh']
        labels = v['label']
        label = v['label_instance']
        mesh_data = EasyDict(
          {'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label,
           'labels': labels})
        m = add_fields_and_dump_model(mesh_data, fields_needed, p_out + '/' + k, dataset_name)
        print('saved model {}'.format(k))
        continue
    else:
      mesh = load_mesh(file, classification=classification)
      mesh_orig = mesh
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
      if add_labels:
        if type(add_labels) is list:
          fn2labels_map = add_labels
        else:
          fn2labels_map = None
        label, labels_orig = get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
      else:
        label = np.zeros((0, ))
      for this_target_n_faces in n_target_faces:
        mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
        save_fn = out_fn + str_to_add
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      m = add_fields_and_dump_model(mesh_data, fields_needed, save_fn, dataset_name)

# ------------------------------------------------------- #

def prepare_modelnet40():
  n_target_faces = [4000]
  labels2use = model_net_labels
  for i, name in tqdm(enumerate(labels2use)):
    for part in ['test', 'train']:
      pin = os.path.expanduser('~') + '/Databases/ModelNet40/' + name + '/' + part + '/'
      p_out = os.path.expanduser('~') + '/mesh_walker/datasets/modelnet40_upsample_4k/'
      prepare_directory_from_scratch('modelnet40', pathname_expansion=pin + '*.off',
                                     p_out=p_out, add_labels='modelnet', n_target_faces=n_target_faces,
                                     fn_prefix=part + '_', verbose=False)


def prepare_s3dis():
  labels2use = s3dis_labels
  for n in ['0','1','2','3','4','5a', '5b', '6']:
    pin = os.path.expanduser('~') + '/Databases/S3DIS/area_' + n + '/3d'
    p_out = os.path.expanduser('~') + '/mesh_walker/datasets/s3dis/area_' + n
    prepare_directory_from_scratch('s3dis', pathname_expansion=pin + '/semantic.obj',
                                   p_out=p_out, add_labels='s3dis', n_target_faces=None,
                                   fn_prefix='area_' + n + '_', verbose=False)

def prepare_shrec17():
  n_target_faces = [2000, 4000, 10000]
  labels2use = shrec17_labels
  for i, name in tqdm(enumerate(labels2use)):
    for part in ['test', 'train']:
      pin = os.path.expanduser('~') + '/Databases/Shrec17/holes/' + part + '/'
      p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed/shrec17/'
      prepare_directory_from_scratch('shrec17', pathname_expansion=pin + '*' + name + '*.off',
                                     p_out=p_out, add_labels='shrec17', n_target_faces=n_target_faces,
                                     fn_prefix=part + '_', verbose=False)


def prepare_cubes(labels2use=cubes_labels,
                  path_in=os.path.expanduser('~') + '/datasets/cubes/',
                  p_out=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/cubes_tmp'):
  dataset_name = 'cubes'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    for part in ['test', 'train']:
      pin = path_in + name + '/' + part + '/'
      prepare_directory_from_scratch(dataset_name, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                                     classification=False)


def prepare_shrec11_from_raw():
  # Prepare labels per model name
  current_label = None
  model_number2label = [-1 for _ in range(600)]
  for line in open(os.path.expanduser('~') + '/datasets/shrec11/evaluation/test.cla'):
    sp_line = line.split(' ')
    if len(sp_line) == 3:
      name = sp_line[0].replace('_test', '')
      if name in shrec11_labels:
        current_label = name
      else:
        raise Exception('?')
    if len(sp_line) == 1 and sp_line[0] != '\n':
      model_number2label[int(sp_line[0])] = shrec11_shape2label[current_label]


  # Prepare npz files
  p_in = os.path.expanduser('~') + '/datasets/shrec11/raw/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/shrec11_raw_1.5k/'
  prepare_directory_from_scratch('shrec11', pathname_expansion=p_in + '*.off',
                                 p_out=p_out, add_labels=model_number2label, n_target_faces=[1500])

  # Prepare split train / test
  change_train_test_split(p_out, 16, 4, '16-04_C')


def prepare_human_body_segmentation():
  dataset_name = 'sig17_seg_benchmark'
  human_seg_path = os.path.expanduser('~') + '/mesh_walker/datasets_raw/sig17_seg_benchmark/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/sig17_seg_benchmark/'

  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  n_target_faces = [1500]
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  for part in ['test', 'train']:
    print('part: ', part)
    path_meshes = human_seg_path + '/meshes/' + part
    seg_path = human_seg_path + '/segs/' + part
    all_fns = []
    for fn in Path(path_meshes).rglob('*.*'):
      all_fns.append(fn)
    for fn in tqdm(all_fns):
      model_name = str(fn)
      if model_name.endswith('.obj') or model_name.endswith('.off') or model_name.endswith('.ply'):
        new_fn = model_name[model_name.find(part) + len(part) + 1:]
        new_fn = new_fn.replace('/', '_')
        new_fn = new_fn.split('.')[-2]
        out_fn = p_out + '/' + part + '__' + new_fn
        mesh = mesh_orig = load_mesh(model_name, classification=False)
        mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
        face_labels = get_sig17_seg_bm_labels(mesh_data, model_name, seg_path)
        labels_orig, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh_data, face_labels)
        if 0: # Show segment borders
          b_vertices = np.where(np.sum(v_labels_fuzzy != 0, axis=1) > 1)[0]
          vertex_colors = np.zeros((mesh_data['vertices'].shape[0],), dtype=np.int)
          vertex_colors[b_vertices] = 1
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=vertex_colors, point_size=2)
        if 0: # Show face labels
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], face_colors=face_labels, show_vertices=False, show_edges=False)
        if 0:
          print(model_name)
          print('min: ', np.min(mesh_data['vertices'], axis=0))
          print('max: ', np.max(mesh_data['vertices'], axis=0))
          cpos = [(-3.5, -0.12, 6.0), (0., 0., 0.1), (0., 1., 0.)]
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=labels_orig, cpos=cpos)
        add_labels = 1
        label = -1
        def calc_face_labels_after_remesh(mesh_orig, mesh, face_labels):
          t_mesh = trimesh.Trimesh(vertices=np.array(mesh_orig.vertices), faces=np.array(mesh_orig.triangles), process=False)

          remeshed_face_labels = []
          for face in mesh.triangles:
            vertices = np.array(mesh.vertices)[face]
            center = np.mean(vertices, axis=0)
            p, d, closest_face = trimesh.proximity.closest_point(t_mesh, [center])
            remeshed_face_labels.append(face_labels[closest_face[0]])
          return remeshed_face_labels

        for this_target_n_faces in n_target_faces:
          mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
          remeshed_face_labels = calc_face_labels_after_remesh(mesh_orig, mesh, face_labels)
          mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
          if 1:
            v_labels, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh_data, remeshed_face_labels)
            mesh_data['labels'] = v_labels
            mesh_data['labels_fuzzy'] = v_labels_fuzzy
            fileds_needed += ['labels_fuzzy']
          if 0:  # Show segment borders
            b_vertices = np.where(np.sum(v_labels_fuzzy != 0, axis=1) > 1)[0]
            vertex_colors = np.zeros((mesh_data['vertices'].shape[0],), dtype=np.int)
            vertex_colors[b_vertices] = 1
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=vertex_colors, point_size=10)
          if 0:  # Show face labels
            utils.visualize_model(np.array(mesh.vertices), np.array(mesh.triangles), face_colors=remeshed_face_labels, show_vertices=False, show_edges=False)
          out_fc_full = out_fn + str_to_add
          if os.path.isfile(out_fc_full + '.npz'):
            continue
          add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
          if 0:
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                                  cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])


def prepare_human_seg_from_meshcnn(dataset_name='human_seg_from_meshcnn', labels2use=coseg_labels,
                  path_in=os.path.expanduser('~') + '/datasets/human_seg/',
                  p_out_root=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/human_seg_tmp_nrmls'):
  p_out = p_out_root + '/'

  for part in ['test', 'train']:
    pin = path_in + '/' + part + '/'
    prepare_directory_from_scratch(dataset_name, pathname_expansion=pin + '*.obj',
                                   p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                                   fix_mesh_fn=fix_mesh_human_body_from_meshcnn)

def prepare_coseg(dataset_name='coseg',
                  path_in=os.path.expanduser('~') + '/datasets/coseg/',
                  p_out_root=os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/coseg_tmp2'):
  for sub_folder in os.listdir(path_in):
    p_out = p_out_root + '/' + sub_folder
    if not os.path.isdir(p_out):
      os.makedirs(p_out + '/' + sub_folder)

    for part in ['test', 'train']:
      pin = path_in + '/' + sub_folder + '/' + part + '/'
      prepare_directory_from_scratch(sub_folder, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf])

# ------------------------------------------------------- #

def map_fns_to_label(path=None, filenames=None):
  lmap = {}
  if path is not None:
    iterate = glob.glob(path + '/*.npz')
  elif filenames is not None:
    iterate = filenames

  for fn in iterate:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    label = int(mesh_data['label'])
    if label not in lmap.keys():
      lmap[label] = []
    if path is None:
      lmap[label].append(fn)
    else:
      lmap[label].append(fn.split('/')[-1])
  return lmap


def change_train_test_split(path, n_train_examples, n_test_examples, split_name):
  np.random.seed()
  fns_lbls_map = map_fns_to_label(path)
  for label, fns_ in fns_lbls_map.items():
    fns = np.random.permutation(fns_)
    assert len(fns) == n_train_examples + n_test_examples
    train_path = path + '/' + split_name + '/train'
    if not os.path.isdir(train_path):
      os.makedirs(train_path)
    test_path = path + '/' + split_name + '/test'
    if not os.path.isdir(test_path):
      os.makedirs(test_path)
    for i, fn in enumerate(fns):
      out_fn = fn.replace('train_', '').replace('test_', '')
      if i < n_train_examples:
        shutil.copy(path + '/' + fn, train_path + '/' + out_fn)
      else:
        shutil.copy(path + '/' + fn, test_path + '/' + out_fn)


# ------------------------------------------------------- #


def prepare_one_dataset(dataset_name, mode):
  dataset_name = dataset_name.lower()
  if dataset_name == 'modelnet40' or dataset_name == 'modelnet':
    prepare_modelnet40()

  if dataset_name == 'shrec17':
    prepare_shrec17()

  if dataset_name == 's3dis':
    prepare_s3dis()

  if dataset_name == 'shrec11':
    pass

  if dataset_name == 'cubes':
    pass

  # Semantic Segmentations
  if dataset_name == 'human_seg':
    if mode == 'from_meshcnn':
      prepare_human_seg_from_meshcnn()
    else:
      prepare_human_body_segmentation()

  if dataset_name == 'coseg':
    pass


def visualize_segmentation_dataset(pathname_expansion):
  cpos = None
  filenames = glob.glob(pathname_expansion)
  while 1:
    fn = np.random.choice(filenames)
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    cpos = utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int), cpos=cpos)


if __name__ == '__main__':
  TEST_FAST = 0
  # utils.config_gpu(False)
  np.random.seed(1)

  dataset_name = 's3dis'   #'s3dis'   #'shrec17'
  mode = None
  if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
  if len(sys.argv) > 2:
    mode = sys.argv[2]

  if dataset_name == 'all':
    for dataset_name_ in ['modelnet40', 'shrec11', 'cubes', 'human_seg', 'coseg', 's3dis']:
      prepare_one_dataset(dataset_name_)
  else:
    prepare_one_dataset(dataset_name, mode)
