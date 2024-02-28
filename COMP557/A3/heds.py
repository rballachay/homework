# Riley Ballachay, 261019324
import numpy as np

class HalfEdge:
	def __init__(self):
		self.n = None # next
		self.o = None # opposite
		self.v = None # vertex
		self.f = None # face
		self.child1 = None # first child half edge
		self.child2 = None # second child half edge
		self.parent = None # parent half edge

	def get_curve_nodes(self):
		# get positions for drawing this half edge with polysope
		nodes = np.zeros((3,3))
		nodes[0] = self.n.n.v.p * 0.90 + self.v.p * 0.05 + self.n.v.p * 0.05
		nodes[1] = self.n.n.v.p * 0.05 + self.v.p * 0.90 + self.n.v.p * 0.05
		nodes[2] = self.n.n.v.p * 0.05 + self.v.p * 0.80 + self.n.v.p * 0.15
		return nodes

class Vertex:
	def __init__(self, p):
		self.p = p # position of the point
		self.n = None # normal of the limit surface
		self.he = None # a half edge that points to this vertex
		self.child = None # child vertex

class Face:
	def __init__(self, he):
		self.he = he

class HEDS:
	def	__init__(self, V, F):
		self.verts = []
		self.faces = []
		if len(V)==0: return
		# TODO: create vertex and face lists along with the half edge data structure
		for v in V:
			_vertex = Vertex(v)
			self.verts.append(_vertex)

		_half_edges = []
		for f in F:
			_face = None
			# just use the last half-edge added to the face
			for u, v in [(self.verts[f[i]],self.verts[f[j]]) for i,j in [(0,1),(1,2),(2,0)]]:
				# this is the half-edge from location u to v
				_half_edge = HalfEdge()

				# use the first half-edge in the face
				if _face is None:
					_face = Face(_half_edge)

				_half_edge.v = v # vertex is destination vertex
				_half_edge.f = _face
				v.he = _half_edge
				_half_edges.append(_half_edge)
			
			self.faces.append(_face)

			# calculate next 
			for u, v in [(self.verts[f[i]],self.verts[f[j]]) for i,j in [(0,1),(1,2),(2,0)]]:
				u.he.n = v.he

		# Populate the dictionary with half edges keyed by their starting vertex
		for half_edge in _half_edges:
			end = half_edge.v

			# filter out the edges that aren't connected to this vertex
			edge_targets = list(filter(lambda he: (he.v==end) and (he!=half_edge) and (half_edge.n.n.v==he.n.v), _half_edges))
			opposite = edge_targets[0].n
			opposite.o = half_edge
			half_edge.o = opposite
		
	def get_even_verts(self):
		# get positions for drawing even vertices with polysope
		if len(self.verts)==0: return []
		if self.verts[0].child is None: return []
		return np.array([v.child.p for v in self.verts])

	def get_odd_verts(self):
		# get positions for drawing odd vertices with polysope
		if len(self.faces)==0: return []
		if self.faces[0].he.child1 is None: return []
		odd_verts = [ [f.he.child1.v.p for f in self.faces], [f.he.n.child1.v.p for f in self.faces], [f.he.n.n.child1.v.p for f in self.faces] ]
		odd_verts = np.array(odd_verts).reshape(-1, 3)
		return odd_verts

	def get_mesh(self):
		# get positions and faces for drawing the mesh with polysope
		for i,v in enumerate(self.verts): v.ix = i # assign an index to each vertex
		V = np.array([v.p for v in self.verts])
		F = np.array([[f.he.v.ix, f.he.n.v.ix, f.he.n.n.v.ix] for f in self.faces])
		return V, F
	
	def get_limit_normals(self):
		# get the limit normals, if they were computed, otherwise returns nothing
		if len(self.verts)==0: return []
		if self.verts[0].n is None: return []
		return np.array([v.n for v in self.verts])