import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


class MeshFPFH(object):
	def __init__(self, mesh, nrings, div=11):
		'''

		:param mesh: open3d object? or .npz with pre-calculated VERTICES normals
		:param nrings:
		'''
		self.mesh = mesh
		self.n_rings = nrings
		self._div = div   # number of bins per angle  (N_angles * div  will be our histogram dimension)

		self.ranges = [(),(),(-np.pi, np.pi)]  # constant range for alpha, phi, theta
		assert nrings >= 1 and nrings < 5  # will be too exhaustive


	def _get_K_rings(self):
		'''
		:return: k_rings, N_points X max(N_ring_neighbors) - for each vertex, return neighbor vertices (with K-edges connectivity). padded.
		'''
		cur_indices = [[i] for i in range(self.mesh.edges.shape[0])]  # the 0-ring neighbors
		rings_indices = [[i] for i in range(self.mesh.edges.shape[0])]
		for _ in range(1, self.n_rings+1):
			cur_indices = [np.unique([self.mesh.edges[i] for i in x if i != -1]) for x in cur_indices]
			for i, cur_ring in enumerate(cur_indices):
				cur_ring = list(cur_ring)
				if cur_ring[0] == -1:
					cur_ring = cur_ring[1:]
				rings_indices[i] = list(np.unique(rings_indices[i] + cur_ring))
		# moving to matrix and padding with -1
		max_inds = max([len(x) for x in rings_indices])
		rings_inds_mat = -np.ones((len(rings_indices), max_inds)).astype(np.int32)
		mask = np.zeros((len(rings_indices), max_inds)).astype(np.bool)
		for i, ring in enumerate(rings_indices):
			rings_inds_mat[i,:len(ring)] = ring
			mask[i, :len(ring)] = 1
		return rings_inds_mat, mask


	def calc_fpfh(self):
		rings, mask = self._get_K_rings()
		rings_distances = []
		spfh = []
		for i, ring in enumerate(rings):
			r = [x for x in ring if x != -1]
			vertices = self.mesh.vertices[r]
			p = vertices[0]  # p-vector of vertices ([v0, v1, ... ,vn])
			# max_ind = np.argwhere(mask[i] == -1)[0]

			# TODO: test if we need to take p-pt or pt-p according to paper (smaller angle from normal to both vectors)

			pt_p = vertices[1:] - p
			pt_p_norm = np.linalg.norm(pt_p, axis=1)
			rings_distances.append(pt_p_norm)
			pt_p = pt_p / np.expand_dims(pt_p_norm + 1e-6, axis=-1)


			normals = np.asarray(self.mesh.vertex_normals[r])
			u = normals[0]   # normal of first point p
			v = np.cross(pt_p, u)
			w = np.cross(v, u)
			nt = normals[1:]

			alpha = np.sum(nt * v, axis=-1)
			phi = np.sum(pt_p * u, axis=-1)
			theta = np.arctan2(np.sum(w*nt, axis=-1), np.sum(nt*u, axis=-1))
			spf = np.stack([alpha, phi, theta], axis=0)
			spfh_hist, bin_edges = self.calc_spfh_hist(spf)
			spfh.append(spfh_hist.flatten())
		# we now have a list of spfhs for each vertex (spfh of SELF.)

		# TODO: calculate FPFH from SPFHS , can decide different ring value that used for spfh calc! for now will be the same
		fpfh = np.zeros_like(np.asarray(spfh))

		# Normalizing rings distances for effective choice of wk (weighting of SPF per neighbor)
		weights = [np.exp(-(ring - np.min(ring)) / (1e-6 + np.min(ring) * (2* np.var((ring - np.min(ring)) / np.min(ring))))) for ring in rings_distances]

		for i, s in enumerate(spfh):
				fpfh[i] = s + np.mean([spfh[k] * weights[i][j] for j, k in enumerate(rings[i,1:]) if k != -1], axis=0)
		return fpfh



	def calc_spfh_hist(self, features):
		spfh_hist = np.zeros((3, self._div))
		bin_edges = np.zeros((3, self._div+1))
		ranges = [(-1, 1), (-1, 1), (-np.pi, np.pi)]
		for i in range(3):
			spfh_hist[i], bin_edges[i] = np.histogram(features[i], bins=self._div, range=ranges[i])

		return spfh_hist, bin_edges


	def calc_thresholds(self):

		"""
		:returns: 3x(div-1) array where each row is a feature's thresholds
		"""
		delta = 2. / self._div
		s1 = np.array([-1 + i * delta for i in range(0, self._div+1)])

		delta = 2. / self._div
		s3 = np.array([-1 + i * delta for i in range(0, self._div+1)])

		delta = (np.pi) / self._div
		s4 = np.array([-np.pi / 2 + i * delta for i in range(0, self._div+1)])

		s = np.array([s1, s3, s4])
		return s



if __name__ == '__main__':
	from dataset import load_model_from_npz
	from easydict import EasyDict
	mesh = load_model_from_npz('/home/ran/mesh_walker/datasets/modelnet40_1k2k4k/test_airplane_0627_simplified_995.npz')
	mesh = EasyDict({'vertices': mesh['vertices'],
									 'faces': mesh['faces'],
									 'v_normals': mesh['vertex_normals'],
									 'edges': mesh['edges']})
	fph = MeshFPFH(mesh, 2)
	s_t = time.time()
	fpfh = fph.calc_fpfh()
	print('{:2.3f} seconds'.format(time.time() - s_t))
	print('')