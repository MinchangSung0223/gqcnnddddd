from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from autolab_core import YamlConfig, Logger
from perception import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                        RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode
import cv2
import os
logger = Logger.get_logger("grabdepth_segmask_gqcnn.py")
config_filename = os.path.join("gqcnn_pj_realsense.yaml")
config = YamlConfig(config_filename)
policy_config = config["policy"]
policy_type = "cem"
policy = CrossEntropyRobustGraspingPolicy(policy_config)

import pybullet as p
import time
import numpy as np
import pybullet_data
from scipy.linalg import null_space
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
import math
pi = math.pi
import cv2
indyEndEffectorIndex = 7
imageWidth  = 640;
imageHeight  = 640;
camera_intr_filename = "realsense.intr"
camera_intr = CameraIntrinsics.load(camera_intr_filename)
def run_gqcnn(depth,seg_mask):
	best_angle = 0;
	best_point = [0,0];
	best_dist = 0;
	depth_im =DepthImage(depth.astype("float32"), frame=camera_intr.frame)
	color_im = ColorImage(np.zeros([imageWidth, imageHeight,3]).astype(np.uint8),
                          frame=camera_intr.frame)
	segmask = BinaryImage(seg_mask)
	rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
	state_gqcnn = RgbdImageState(rgbd_im, camera_intr, segmask=segmask) 
	policy_start = time.time()
	q_value = -1
	try:
		action = policy(state_gqcnn)
		best_point = [action.grasp.center[0],action.grasp.center[1]];
		best_angle = float(action.grasp.angle)*180/3.141592
		q_value = action.q_value
		print("inner :       ",action.q_value)

				
	except Exception as inst:
		print(inst)

	return q_value




def getCameraImageEEF(pos,orn):
	#print(eef_pose)
	com_p = pos
	com_o = orn
	rot_matrix = p.getMatrixFromQuaternion(com_o)
	rot_matrix = np.array(rot_matrix).reshape(3, 3)
	# Initial vectors
	init_camera_vector = (0, 0, 1) # z-axis
	init_up_vector = (0, 1, 0) # y-axis
	# Rotated vectors
	camera_vector = rot_matrix.dot(init_camera_vector)
	up_vector = rot_matrix.dot(init_up_vector)
	view_matrix = p.computeViewMatrix([com_p[0],com_p[1],com_p[2]],[0,0,0], up_vector)
	fov = 60
	aspect = imageWidth/imageHeight
	near = 0.01
	far = 1000
	angle = 0.0;


	projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
	images = p.getCameraImage(imageWidth,
					imageHeight,
					view_matrix,
					projection_matrix,
					shadow=True,
					renderer=p.ER_BULLET_HARDWARE_OPENGL)
	return images,view_matrix
def getMatrixfromEEf(eef_pose):
	Tbe = np.eye(4)
	eef_p = np.array(eef_pose[0])
	eef_o = np.array(eef_pose[1])
	#print(eef_o)
	R = np.reshape(p.getMatrixFromQuaternion(eef_o),[3,3])
	R2 = []
	Tbe[0:3,0:3] = R
	Tbe[0,3] = eef_p[0]
	Tbe[1,3] = eef_p[1]
	Tbe[2,3] = eef_p[2]
	return Tbe
def sample_sphere(num_samples,dist):
	""" sample angles from the sphere
	reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
	"""
	begin_elevation = 0
	ratio = (begin_elevation + 90) / 180
	num_points = int(num_samples // (1 - ratio))
	phi = (np.sqrt(5) - 1.0) / 2.
	x_list = []
	y_list = []
	z_list = []
	azi_list = []
	ele_list = []
	for n in range(num_points - num_samples, num_points):
		z = 2. * n / num_points - 1.
		azi = 2 * np.pi * n * phi % (2 * np.pi)
		ele = np.arcsin(z)
		azi_list.append(azi)
		ele_list.append(ele)
		x = dist*math.cos(azi)*math.cos(ele)
		y = dist*math.sin(azi)*math.cos(ele)
		z = dist*math.sin(ele)
		x_list.append(x)
		y_list.append(y)
		z_list.append(z)

	
	return np.array(x_list), np.array(y_list), np.array(z_list), np.array(azi_list), np.array(ele_list)

def printHomogeneous(T,print_str):


	print_str = " ------------------"+print_str+"------------------"
	print(print_str)
	#print(T)
	float_point = 3
	print("|\t"+"   "+"\t"+"  R"+"\t"+"   "+"\t\t|")
	print("|\t"+str("{:.3f}".format(np.around(T[0,0],float_point)))+"\t"+str("{:.3f}".format(np.around(T[0,1],float_point)))+"\t"+str("{:.3f}".format(np.around(T[0,2],float_point)))+"\t\t|")
	print("|\t"+str("{:.3f}".format(np.around(T[1,0],float_point)))+"\t"+str("{:.3f}".format(np.around(T[1,1],float_point)))+"\t"+str("{:.3f}".format(np.around(T[1,2],float_point)))+"\t\t|")
	print("|\t"+str("{:.3f}".format(np.around(T[2,0],float_point)))+"\t"+str("{:.3f}".format(np.around(T[2,1],float_point)))+"\t"+str("{:.3f}".format(np.around(T[2,2],float_point)))+"\t\t|")
	print("|\t"+"  x"+"\t"+"  y"+"\t"+"  z"+"\t\t|")
	print("|\t"+str("{:.3f}".format(np.around(T[0,3],float_point)))+"\t"+str("{:.3f}".format(np.around(T[1,3],float_point)))+"\t"+str("{:.3f}".format(np.around(T[2,3],float_point)))+"\t\t|")
	end_str=""
	for i in range(len(print_str)):
		end_str = end_str+"-"
	print(end_str)
	lineLen = 0.1
	pos = [T[0,3],T[1,3],T[2,3]]
	dir0 = [T[0,0],T[1,0],T[2,0]]
	dir1 = [T[0,1],T[1,1],T[2,1]]
	dir2 = [T[0,2],T[1,2],T[2,2]]

	toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
	toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
	toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
	p.addUserDebugLine(pos, toX, [1, 0, 0], 2,0.1)
	p.addUserDebugLine(pos, toY, [0, 1, 0], 2,0.1)
	p.addUserDebugLine(pos, toZ, [0, 0, 1], 2,0.1)


if __name__ == "__main__":
	clid = p.connect(p.SHARED_MEMORY)
	if (clid < 0):
		p.connect(p.GUI)
		#p.connect(p.SHARED_MEMORY_GUI)
	p.setGravity(0, 0, -0)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	obj = p.loadURDF("object/model_normalized_convex.urdf", [0, 0, 0.00],p.getQuaternionFromEuler([0, 0, 0]))
	p.loadURDF("plane.urdf", [0, 0, -5],p.getQuaternionFromEuler([0,0,0]))

	aabb = p.getAABB(obj)
	aabbMin = np.array(aabb[0])
	aabbMax = np.array(aabb[1])
	aabbRange = abs(aabbMax - aabbMin)
	minCamDistance = max(aabbRange)*2.5
	print(aabbRange)
	print(minCamDistance)
	t= 0
	N = 100;
	x,y,z,azi,ele = sample_sphere(N,minCamDistance)
	print(x)
	for i in range(0,N):
		pos = [x[i],y[i],z[i]]
		p0 = np.array([0,0,0,1])
		p0_x = np.array([0.1,0,0,1])
		p0_y = np.array([0,0.1,0,1])
		p0_z = np.array([0,0,0.1,1])




		orn = p.getQuaternionFromEuler([0,pi,0])
		images,view_matrix = getCameraImageEEF(pos,orn)
		rgb = images[2];

		depth = np.reshape(images[3],[imageWidth,imageHeight]);
		seg_mask = (images[4]+1)
		seg_mask[seg_mask>1] =0
		seg_mask = np.reshape(np.uint8(seg_mask*255),[imageWidth,imageHeight]);
		max_q_value = 0
		k_list = []
		q_list = []
		for j in range(0,100):
			k =0.605
			depth_temp = depth*k
			q_value= run_gqcnn(depth_temp,seg_mask);
			print("\n\n\n")
			print("k: ",k,"Q Value = ", q_value)
			print("\n\n\n")
			q_list.append(q_value);
			k_list.append(k);
		print(np.max(np.array(q_list)))
		print(np.argmax(np.array(q_list)))
		print(np.array(k_list))
		time.sleep(100)
		
		p.stepSimulation()
	time.sleep(100)
