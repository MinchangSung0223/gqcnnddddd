from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy.ctypeslib import ndpointer
import argparse
import json
import os
import time
import numpy as np
import math
import threading
import time
from autolab_core import YamlConfig, Logger
from perception import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                        RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode
logger = Logger.get_logger("grabdepth_segmask_gqcnn.py")
#from lib.config import cfg, args
#from lib.networks import make_network
#from lib.datasets import make_data_loader
#from lib.utils.net_utils import load_network
from numpy.ctypeslib import ndpointer
best_center=[0 ,0]
center=[0 ,0]
prev_q_value=0
q_value=0
prev_depth=0
#-------------------------------------gqcnn-----------------------------------
config_filename = os.path.join("gqcnn_pj_realsense.yaml")
config = YamlConfig(config_filename)
policy_config = config["policy"]
policy_type = "cem"
policy = CrossEntropyRobustGraspingPolicy(policy_config)

def convert_depth_frame_to_pointcloud(u,v,z):
	camera_intrinsics ={"fx":608.13312805,"ppx": 324.970655,"fy":612.61549006,"ppy":242.1411}
	x = (u - camera_intrinsics["ppx"])/camera_intrinsics["fx"]
	y = (v - camera_intrinsics["ppy"])/camera_intrinsics["fy"]
	z = z ;
	return x,y,z
def run_gqcnn():
   global depth_im
   global color_im
   global segmask
   global policy
   global prev_q_value
   global prev_depth
   global toggle
   global t
   global x
   global y
   global z
   global best_angle
   global depth_raw
   global num_find_loc
   global state
   global loc_count
   global no_find
   global center
   global q_value
   global angle
   global no_valid_grasp_count
   global no_valid_move_y 
   global depth_frame
   global frames
   no_valid_grasp_count = 0;
   time.sleep(1)
   depth_frame = frames.get_depth_frame()
   depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
   best_angle = 0
   x=0.0
   y=0.5
   action =0
   num_find_loc=0
   while 1:
     try:
       #depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
       if "input_images" in policy_config["vis"] and policy_config["vis"][
            "input_images"]:
             plt.pause(0.001)
             #pass
       # Create state.
       rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
       state_gqcnn = RgbdImageState(rgbd_im, camera_intr, segmask=segmask) 

       # Query policy.
       policy_start = time.time()
       try:
            action = policy(state_gqcnn)
            logger.info("Planning took %.3f sec" % (time.time() - policy_start))
            no_valid_grasp_count =  0;
       except Exception as inst:
          print(inst)
          no_valid_grasp_count = no_valid_grasp_count+1;
          time.sleep(0.3)

       if policy_config["vis"]["final_grasp"]:
        center[0] = action.grasp.center[0]
        center[1] = action.grasp.center[1]
        depth = depth_frame.get_distance(int(center[0]), int(center[1]))
       #depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(center[0]), int(center[1])], depth)
        #print(depth_point)




        q_value = action.q_value
        angle = float(action.grasp.angle)*180/3.141592
        x,y,z = convert_depth_frame_to_pointcloud(int(center[0]), int(center[1]),depth)
        print("center : \t"+str(action.grasp.center))
        print("angle : \t"+str(action.grasp.angle)) 
        print("Depth : \t"+str(action.grasp.depth)) 

        print("\n\n\n\n\n\n\nQ_value : \t" +str(action.q_value))
        if(prev_q_value<action.q_value):
           prev_q_value = action.q_value
           prev_depth = action.grasp.depth
           best_center[0] =action.grasp.center[0]
           best_center[1] =action.grasp.center[1]
           depth = depth_frame.get_distance(int(best_center[0]), int(best_center[1]))
           
           #depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(best_center[0]), int(best_center[1])], depth)
           #print(depth_point)
           #convert_data_ = convert_2d_2d(int(best_center[0]),int(best_center[1]))
           #convert_data=convert_2d_3d(int(convert_data_[0]),int(convert_data_[1]))
           x,y,z = convert_depth_frame_to_pointcloud(int(best_center[0]), int(best_center[1]),depth)

           best_angle = action.grasp.angle
           best_angle = float(best_angle)*180/3.141592
           
           print("gqcnn_best_center : \t"+str(x)+","+str(y))
           print("best_angle : \t"+str(best_angle))
           print("\n\n\n\n\n\n\nbest_Q_value : \t" +str(action.q_value))
           print("XYZ : \t" +str(x)+"\t"+str(y)+"\t"+str(z))

        num_find_loc = num_find_loc+1
        if num_find_loc >5:
           prev_q_value = 0
           x=0
           y=0
           z=0
           num_find_loc=0 
           best_angle = 0
           best_center[0] = 0
           best_center[1] = 0
        time.sleep(0.01);
     except Exception as inst:
       print(inst)
       time.sleep(1);
       pass

#-----------------------------------------------------------------------------

#import matplotlib.pyplot as plt
import ctypes
import numpy as np
import cv2
import os
import threading
import tqdm
import torch
#from lib.visualizers import make_visualizer
import pycocotools.coco as coco
#from lib.datasets import make_data_loader
#from lib.evaluators import make_evaluator
import tqdm
import torch
#from lib.utils.pvnet import pvnet_pose_utils
#from lib.networks import make_network
import matplotlib.patches as patches
#from lib.utils.net_utils import load_network
from ctypes import cdll
import pyrealsense2 as rs
#---------------------------------------realsense---------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

camera_intr_filename = "realsense.intr"
camera_intr = CameraIntrinsics.load(camera_intr_filename)
def detect_img():
    global color_img
    global mask

    global depth_im
    global color_im
    global segmask
    global center
    global angle
    global best_angle
    global prev_q_value
    global best_center
    global depth_frame
    global frames
    def nothing(x):
       pass
    mask = np.ones((640,480),dtype=np.uint8)*255
    cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rgb", 1280,720)
    cv2.createTrackbar('W', "rgb", 0, 100, nothing)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask", 640,480)
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("depth", 1000,1000)
    line_arr1 = [0, 1, 3, 2, 0, 4, 6, 2]
    line_arr2 = [5, 4, 6, 7, 5, 1, 3, 7]

    divide_value = 65535
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    while 1:
     try:
      frames = pipeline.wait_for_frames()
      w = cv2.getTrackbarPos('W',"rgb")
      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()

      if not depth_frame or not color_frame:
            continue
      depth_to_color_img = np.asanyarray(depth_frame.get_data())/divide_value
      color_img = np.asanyarray(color_frame.get_data())
      #print(depth_to_color_img)
      depth_im =DepthImage(depth_to_color_img.astype("float32"), frame=camera_intr.frame)
      color_im = ColorImage(np.zeros([480, 640,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)
      seg_mask =cv2.resize(mask.copy(),(640,480),interpolation=cv2.INTER_CUBIC)
      segmask = BinaryImage(seg_mask)
      show_img = cv2.resize(color_img.copy(),(640,480),interpolation=cv2.INTER_CUBIC)
      try:
         ret, img_binary = cv2.threshold(mask, 127, 255, 0)
         contours, color_hierachy =cv2.findContours(img_binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
         c0 = np.reshape(contours[0],(-1,2))
         show_img = cv2.drawContours(np.uint8(show_img), contours, -1, (0,255,0), 1)
      except:
         pass
      show_img_resize = cv2.resize( show_img.copy(),(640,480),interpolation=cv2.INTER_CUBIC)
      try:
         r=50
         show_img_resize = cv2.line(show_img_resize, (int(best_center[0]-r*math.cos(best_angle/180*math.pi)),int(best_center[1]-r*math.sin(best_angle/180*math.pi))), (int(best_center[0]+r*math.cos(best_angle/180*math.pi)),int(best_center[1]+r*math.sin(best_angle/180*math.pi))), (0,0,255), 5)
         show_img_resize = cv2.line(show_img_resize, (int(best_center[0]),int(best_center[1])), (int(best_center[0]+1),int(best_center[1])) , (0,136,255), 7)
         show_img_resize = cv2.line(show_img_resize, (int(center[0]-r*math.cos(angle/180*math.pi)),int(center[1]-r*math.sin(angle/180*math.pi))), (int(center[0]+r*math.cos(angle/180*math.pi)),int(center[1]+r*math.sin(angle/180*math.pi))), (255,100,255), 5)
         show_img_resize = cv2.line(show_img_resize, (int(center[0]),int(center[1])), (int(center[0]+1),int(center[1])) , (100,136,255), 7)
       
      except:
       	 pass
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_to_color_img*1000, alpha=2), 1)
      cv2.imshow("rgb",show_img_resize)
      cv2.imshow("depth",depth_colormap)
      cv2.imshow("mask",seg_mask)
      k = cv2.waitKey(5) & 0xFF
      if k == ord('s'):
         cv2.destroyWindow("rgb")
     except Exception as inst:
      print(inst)
      pass
    end()



def do_Robot():
	READY=0;
	FIND=1;
	GRASP=2;
	global state
	global x
	global y
	global z
		
	state = READY;
	time.sleep(1)
	while(1):
		if state==READY:
			print("READY STATE")
			try:
				print("X:",x," | Y:",y," | Z:",z)
			except:
				print("No DATA")
		elif state==FIND:
			print("FIND STATE")
		elif state==GRASP:
			print("GRASP STATE")
		time.sleep(0.1);

if __name__ == '__main__':
    t1 = threading.Thread(target=detect_img)
    t1.start()

    t2 = threading.Thread(target=do_Robot)
    t2.start()


    t3 = threading.Thread(target=run_gqcnn)
    t3.start()



