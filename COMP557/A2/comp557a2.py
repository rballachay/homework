"""
Name: Riley Ballachay
Student Number: 261019324
"""
import moderngl_window as mglw
import moderngl as mgl
import numpy as np
from pyrr import matrix44
import random
from scipy.spatial.transform import Rotation

rotation_type = [ 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', 'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ', 'RL', 'QL', 'QS', 'QLN', 'QLNF', 'QSF','A','B' ]

# dict mapping letters x y z to colors
letter_colors = { 
            'X': np.array((1,0,0),dtype='f4'), 
            'Y': np.array((0,1,0),dtype='f4'), 
            'Z': np.array((0,0,1),dtype='f4'),
            'R': np.array((.3,.7,0),dtype='f4'), 
            'Q': np.array((0,.7,.7),dtype='f4'), 
            'L': np.array((.7,0,.7),dtype='f4'), 
            'S': np.array((.7,.3,0),dtype='f4'),
            'N': np.array((.7,.3,.3),dtype='f4'),  
            'F': np.array((.3,.7,.3),dtype='f4'),  
            'A': np.array((.7,.7,.7),dtype='f4'), 
            'B': np.array((.7,.7,.7),dtype='f4')  
            }

def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Ensure quaternions are unit quaternions
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_1 = np.sin(theta_0 - theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q1) + (s1 * q2)

def rotation_interpolation(q1, q2, t, method):
    """Interpolate rotation between two quaternions using XYZ rotation sequence."""
    # Convert quaternions to rotation matrices
    r1 = quat_to_R(q1)
    r2 = quat_to_R(q2)

    # Convert rotation matrices to XYZ Euler angles
    euler1 = Rotation.from_matrix(r1).as_euler(method.lower())
    euler2 = Rotation.from_matrix(r2).as_euler(method.lower())

    # Interpolate Euler angles
    interpolated_euler = euler1 + t * (euler2 - euler1)

    # Convert interpolated Euler angles back to rotation matrix
    interpolated_r = Rotation.from_euler(method.lower(), interpolated_euler).as_matrix()

    return interpolated_r

def normalize(q):
    return q/np.linalg.norm(q)

def rand_unit_quaternion():
    q = np.array([random.gauss(0, 1) for i in range(4)])    
    return normalize(q)

def quaternion_random_axis_angle(angle): 
    axis = np.array([random.gauss(0, 1) for i in range(3)])
    axis = axis/np.linalg.norm(axis)
    return np.append( np.cos(angle/2), np.sin(angle/2)*axis)

def rand_180_quaternion():
    return quaternion_random_axis_angle(np.pi)

def quaternion_multiply(q1, q2):
    q1q2 = np.zeros(4)
    q1q2[0] = q1[0]*q2[0] - np.dot(q1[1:],q2[1:])
    q1q2[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:],q2[1:])
    return q1q2

def quat_to_R(q):
    return np.array([[1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                        [2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
                        [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)]])

def rotate_x(matrix, angle=90):
    # Convert angle to radians
    theta = np.radians(angle)
    
    # Define rotation matrix about y-axis
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ],dtype='f4')
    
    # Multiply original matrix by rotation matrix
    rotated_matrix = np.dot(matrix, rotation_matrix)
    
    return rotated_matrix

def rotate_y(matrix, angle=180):
    # Convert angle to radians
    theta = np.radians(angle)
    
    # Define rotation matrix about y-axis
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ],dtype='f4')
    
    # Multiply original matrix by rotation matrix
    rotated_matrix = np.dot(matrix, rotation_matrix)
    
    return rotated_matrix


class Body:
	
    def __init__(self,pos,rotorder,prog,vao):
        self.rotorder = rotorder   
        self.pos = pos
        self.prog = prog
        self.vao = vao
        self.colors = letter_colors

        # two quaternions, cannot have any more
        self.rotations = np.zeros((4,2),dtype='f4')
        
    def set_rotation(self, q, i):
        if i>1:
            raise Exception("cannot provide i>1 when setting rotation")
        
        # set rotation
        self.rotations[:,i] = q

    def render(self, t):
        # t is a float between 0 and 1, 
        ## TODO compute the interpolation between target orientations and use it to set the rotation
        ## TODO draw the label for the specified rotaiton type

        M = np.eye(4,dtype='f4')
        M[0:3,3] = self.pos

        M_monkey = self._rotate_monkey(M, t)

        if self.rotorder is None:
            M_monkey = rotate_y(M_monkey)
        
        self.prog['M'].write( M_monkey.T.flatten() )

        self.prog['enable_lighting'] = True
        self.prog['ambient_intensity'] = 0.3
        self.prog['light_position'] = (10,30,20)
        self.prog['diffuse_color']  = (0.5,0.5,0.5)
        self.vao['monkey'].render()

        if self.rotorder is not None:
            M = rotate_x(M)

            # change the starting displacement based on how many letters there are 
            M[0,3]+=(-1.5 + ((4-len(self.rotorder))*0.5) + (len(self.rotorder)-3)*.25)

            # offset 1.5 units below the monkey head
            M[1,3]-=1.5

            for letter in self.rotorder:
                M[0,3]+=.5
                self.prog['M'].write( M.T.flatten() ) # transpose and flatten to get in Opengl Column-majfor format
                self.prog['diffuse_color'] = self.colors.get(letter) 
                self.vao[letter].render()

    def _rotate_monkey(self, M, t):
        rotation_transform = np.eye(4,dtype='f4')
        # select 
        if self.rotorder is None:
            rotation = np.eye(3, dtype='f4')
        elif any([i in self.rotorder for i in ['X','Y','Z']]):
            rotation = rotation_interpolation(self.rotations[:,0],self.rotations[:,1],t,self.rotorder)
        elif self.rotorder=='RL':
            r1 = quat_to_R(self.rotations[:,0])
            r2 = quat_to_R(self.rotations[:,1])
            rotation = (1 - t) * r1 + t * r2
        elif self.rotorder=='QL':
            quat = (1 - t) * self.rotations[:,0] + t * self.rotations[:,1]
            rotation = quat_to_R(quat)
        elif self.rotorder=='QS':
            quat = slerp(self.rotations[:,0],self.rotations[:,1],t)
            rotation = quat_to_R(quat)
        elif self.rotorder=='QLN':
            quat = (1 - t) * self.rotations[:,0] + t * self.rotations[:,1]
            quat = normalize(quat)
            rotation = quat_to_R(quat)
        elif self.rotorder=='QLNF':
            q1 = self.rotations[:,0]
            q2 = self.rotations[:,1]
            dot_product = np.dot(q1, q2)
            if dot_product < 0:
                q1 = -q1
            quat = (1 - t) * q1 + t * q2
            quat = normalize(quat)
            rotation = quat_to_R(quat)
        elif self.rotorder=='QSF':
            q1 = self.rotations[:,0]
            q2 = self.rotations[:,1]
            dot_product = np.dot(q1, q2)
            if dot_product < 0:
                q1 = -q1
            quat = slerp(q1,q2,t)
            rotation = quat_to_R(quat)
        elif self.rotorder=='A':
            quat = self.rotations[:,0]
            rotation = quat_to_R(quat)
        elif self.rotorder=='B':
            quat = self.rotations[:,1]
            rotation = quat_to_R(quat)
        else:
            raise Exception(f"Rotorder of {self.rotorder} not accepted")

        rotation_transform[:3,:3] = rotation
        M_monkey = np.dot(M, rotation_transform)
        return M_monkey


class HelloWorld(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Hello World"
    window_size = (1280, 720)
    aspect_ratio = 16.0 / 9.0
    resizable = True
    resource_dir = 'data'

    def setup_wire_box(self):
        # create cube vertices
        vertices = np.array([
            -1.0, -1.0, -1.0,  
             1.0, -1.0, -1.0,
             1.0,  1.0, -1.0, 
            -1.0,  1.0, -1.0,
            -1.0, -1.0,  1.0, 
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0, 
            -1.0,  1.0,  1.0,
        ], dtype='f4')
        # create cube edges
        indices = np.array([ 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7], dtype='i4')
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        ibo= self.ctx.buffer(indices.astype("i4").tobytes())
        # note that we can provide nothing to the normal attribute, as we will ingore it with the lighting disabled
        self.cube_vao = self.ctx.vertex_array(self.prog, [(vbo, '3f', 'in_position')], index_buffer=ibo, mode=mgl.LINES)

    def draw_wire_box(self):
        self.prog['enable_lighting'] = False
        M=np.eye(4,dtype='f4')

        V_inv = np.linalg.inv(self.V1)
        P_inv = np.linalg.inv(self.P1)

        self.prog['M'].write( (P_inv@V_inv@M.T).flatten() )
        self.cube_vao.render()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        random.seed(0) # set random seed for deterministic reproducibility
        self.prog = self.ctx.program( 
            vertex_shader =open('glsl/vert.glsl').read(), 
            fragment_shader = open('glsl/frag.glsl').read() )

        # load obj files for drawing the monkey and letters
        self.scene = {}
        self.vao = {}
        for a in ['monkey','X','Y','Z','R','L','N','Q','S','F','A','B']:
            self.scene[a] = self.load_scene(a+".obj")
            self.vao[a] = self.scene[a].root_nodes[0].mesh.vao.instance(self.prog)  
               
        self.setup_wire_box()

        # setup a grid of bodies, nicely spaced for viewing
        self.bodies = []

        # add monkey at origin of perspective transformation
        self.bodies.append( Body( np.array([0,0,40]), None, self.prog, self.vao ) )
        for i in range(len(rotation_type)):
            c = 4*((i % 5) - 2)
            r = 4*(-(i // 5) + 1.75)
            self.bodies.append( Body( np.array([c,r,0]), rotation_type[i], self.prog, self.vao ) )

        # initialize the target orientations
        self.A = np.array([1,0,0,0])
        self.set_new_rotations(self.A,0)
        self.B = np.array([1,0,0,0])
        self.set_new_rotations(self.B,1)

        # Setup the primary and secondary viewing and projection matrices        
        self.V1 = matrix44.create_look_at(eye=(0, 0, 40), target=(0, 0, 0), up=(0, 1, 0), dtype='f4')
        self.P1 = matrix44.create_perspective_projection(25.0, self.aspect_ratio, 10, 45, dtype='f4')
        self.V2 = matrix44.create_look_at(eye=(30, 10, 55), target=(0, 0, 20), up=(0, 1, 0), dtype='f4')
        self.P2 = matrix44.create_perspective_projection(40.0, self.aspect_ratio, 10, 100.0, dtype='f4')
        self.V_target = self.V1
        self.P_target = self.P1
        self.V_current = self.V1.copy()
        self.P_current = self.P1.copy()

    def set_new_rotations(self, target, i): 
        for b in self.bodies: b.set_rotation(target,i)
        
    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.A:                
                self.A = rand_unit_quaternion()
                self.set_new_rotations( self.A, 0 )
            if key == self.wnd.keys.B:                
                self.B = rand_unit_quaternion()
                self.set_new_rotations( self.B, 1 )
            if key == self.wnd.keys.Z:
                q = quaternion_random_axis_angle( np.pi/180*3 )
                q = quaternion_multiply( q, self.A )
                self.B = -q
                self.set_new_rotations( self.B, 1 )
            if key == self.wnd.keys.X:                
                q = quaternion_random_axis_angle( np.pi )
                self.B = quaternion_multiply( q, self.A )
                self.set_new_rotations( self.B, 1 )
            if key == self.wnd.keys.I:                
                self.A = np.array([1,0,0,0])
                self.B = np.array([1,0,0,0])
                self.set_new_rotations( self.A, 0 )
                self.set_new_rotations( self.B, 1 )
            elif key == self.wnd.keys.NUMBER_1:
                self.V_target = self.V1
                self.P_target = self.P1
            elif key == self.wnd.keys.NUMBER_2:
                self.V_target = self.V2
                self.P_target = self.P2
               
    def render(self, time, frame_time):
        self.ctx.clear(0,0,0)
        self.ctx.enable(mgl.DEPTH_TEST)

        # Interpolate the current and target viewing and projection matrices
        self.V_current = self.V_current * 0.9 + self.V_target*0.1
        self.P_current = self.P_current * 0.9 + self.P_target*0.1
        self.prog['P'].write( self.P_current )
        self.prog['V'].write( self.V_current )

        time_mod_4 = time%4
        if time_mod_4 < 1: t = time_mod_4
        elif time_mod_4 < 2: t = 1
        elif time_mod_4 <3: t = 3 - time_mod_4
        else: t = 0

        for b in self.bodies:
            b.render(t)

        ## TODO you'll also need some code to draw the viewer and frustum!
        self.draw_wire_box()
    
HelloWorld.run()