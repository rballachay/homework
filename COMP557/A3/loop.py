# Riley Ballachay, 261019324
import numpy as np
from heds import *

def loop_subdivide(heds):
	heds.child = HEDS([],[]) # create an new empty half edge data structure
	for v in heds.verts:
		# TODO subdivide the vertex, and append to heds.child.verts
		v_new = subdivide_vertex(v.he)
		v.child = v_new
		heds.child.verts.append(v_new)

	for f in heds.faces:
		# TODO subdivide edges, if not already done, add new vertex to list
		he = f.he
		for _ in range(3):
			child1, child2, v = subdivide_edge(he)
			he.child1 = child1
			he.child1.v = v
			he.child2 = child2
			he.v.child.he=child2
			heds.child.verts.append(v)
			he=he.n

	for f in heds.faces:
		he = f.he
		for _ in range(3):
			if he.o is not None:
				he.child1.o = he.o.child2
				he.child2.o = he.o.child1
			he=he.n

	for f in heds.faces:
		# TODO connect new vertices to form new faces, and append to heds.child.faces
		
		half_edges_2 = []

		he = f.he
		for _ in range(3):
			# this is the first vertex of the new edge
			v1 = he.child1.v
			v2 = he.n.n.child1.v

			he_1 = HalfEdge()
			new_face = Face(he_1)
			he_1.v = v2
			v2.he = he_1
			
			he_1.f = new_face
			he.child1.f = new_face
			he_1.n = he.n.n.child2
			he.n.n.child2.n = he.child1
			he.n.n.child2.f = new_face
			he.child1.n = he_1
			he_1.parent = he

			# i have no idea why these need to be set like this
			he.child1.v.he = he_1

			he_2 = HalfEdge()
			he_2.v = v1

			# set opposites
			he_2.o = he_1
			he_1.o = he_2

			# i have no idea why these need to be set like this
			he.child1.v.he=he_2

			half_edges_2.append(he_2)

			# append random half edge from face
			heds.child.faces.append(new_face)
			he=he.n
		
		final_face = Face(he_2)
		half_edges_2[0].n = half_edges_2[1] 
		half_edges_2[0].f = final_face
		half_edges_2[1].n = half_edges_2[2] 
		half_edges_2[1].f = final_face
		half_edges_2[2].n = half_edges_2[0] 
		half_edges_2[2].f = final_face
		heds.child.faces.append(final_face)

	for v in heds.child.verts:
		n = compute_limit_normal( v )
		v.n = n

	return heds.child        

def subdivide_vertex( he ):
	boundary=False
	# TODO subdivide the vertex, and append to heds.child.verts
	vertexes = {'center':he.v, 'outside':[]}
	if he.o is not None:
		vertexes['outside'].append(he.o.v)
	
	he_new = he.n
	# get all the edges incident to this vertex
	while he_new.o!=he:
		vertexes['outside'].append(he_new.v)

		if he_new.o is None:
			boundary=True
			break
		he_new=he_new.o
		he_new = he_new.n

	if boundary is True:
		he_new = he

		# this points to the source vertex for the current vertex
		while he_new.o is not None:
			he_new = he_new.o.n.n
		
		p = (3/4)*vertexes['center'].p + (1/8)*vertexes['outside'][-1].p + (1/8)*he_new.n.n.v.p
		
	else:
		degree = len(vertexes['outside'])
		beta = 3/(8*degree) if degree>3 else 3/16
		#beta = (1/degree)*(5/8-(3/8+(1/4)*np.cos(2*np.pi/degree))**2)

		center = (1-degree*beta)*vertexes['center'].p
		outside = np.sum([beta*v.p for v in vertexes['outside']],axis=0)
		p=center+outside
	
	
	return Vertex(p)

def subdivide_edge( he ):
	end_v = he.v.p
	start_v = he.n.n.v.p
	if he.o is None:
		p = (1/2)*start_v+(1/2)*end_v
	else:
		top_v = he.n.v.p
		bottom_v = he.o.n.v.p
		p = (3/8)*start_v+(3/8)*end_v+(1/8)*top_v+(1/8)*bottom_v
	
	v = Vertex(p)

	child1=HalfEdge()
	child1.v = v
	child1.parent = he
	v.he=child1

	child2=HalfEdge()
	child2.v = he.v.child
	child2.parent = he
	return child1, child2, v

def compute_limit_normal( v ):
	# TODO: given Vertex v, compute and set the limit normal 
	boundary=False
	he = v.he

	vertices = []

	he_new = he.n
	while he_new.o!=he:
		vertices.append(he_new.v)
		if he_new.o is None:
			boundary=True
			break
		he_new=he_new.o
		he_new = he_new.n

	if not boundary:
		vertices.insert(0, he.o.v)
		k = len(vertices)
		t1 = np.sum([np.cos(2*np.pi*i/k)*vertices[i].p for i in range(k)],axis=0)
		t2 = np.sum([np.sin(2*np.pi*i/k)*vertices[i].p for i in range(k)],axis=0)
	else:
		he_new = he
		vertices.insert(0,he.n.n.v)
		while he_new.o is not None:
			he_new = he_new.o.n.n
			vertices.insert(0,he_new.n.n.v)

		k = len(vertices)
		t2 = vertices[0].p-vertices[1].p

		if k==2:
			t1 = vertices[0].p+vertices[1].p-2*v.p
		elif k==3:
			t1 = vertices[2].p - vertices[1].p
		elif k>=4:
			theta = np.pi/(k-1)
			
			t1 = np.sin(theta)*(vertices[0].p+vertices[-1].p) + (2*np.cos(theta)-2)*np.sum([np.sin(theta*i)*vertices[i].p for i in range(1,k-1)],axis=0)

			if k==5:
				t2=t1
				t1=vertices[0].p-vertices[1].p


	cross = np.cross(t2,t1)
	return  cross / np.sqrt(np.sum(cross**2))