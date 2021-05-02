from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
config_filename = os.path.join("gqcnn_pj_kinect.yaml")
config = YamlConfig(config_filename)
policy_config = config["policy"]
policy_type = "cem"
policy = CrossEntropyRobustGraspingPolicy(policy_config)
camera_intr = CameraIntrinsics.load("kinect.intr")
WIDTH = 512
HEIGHT = 512

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
   no_valid_grasp_count = 0;
   num_find_loc=0;
   best_angle = 0
   x=0.0
   y=0.5
   z = 0
   time.sleep(5);
   while 1:
     try:
       print("GQCNN IS RUNNING")
       #depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
       if "input_images" in policy_config["vis"] and policy_config["vis"][
            "input_images"]:
             plt.pause(0.001)
             #pass
       # Create state.
       print("GQCNN IS RUNNING2")
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
       print("GQCNN IS RUNNING3")
       if policy_config["vis"]["final_grasp"]:
        print("GQCNN IS RUNNING4")
        center[0] = action.grasp.center[0]
        center[1] = action.grasp.center[1]
        q_value = action.q_value
        angle = float(action.grasp.angle)*180/3.141592
        print("center : \t"+str(action.grasp.center))
        print("angle : \t"+str(action.grasp.angle)) 
        print("Depth : \t"+str(action.grasp.depth)) 
        print("XYZ : \t" +str(x)+" , "+str(y)+" , "+str(z))
        print("\n\n\n\n\n\n\nQ_value : \t" +str(action.q_value))
        if(prev_q_value<action.q_value):
           prev_q_value = action.q_value
           prev_depth = action.grasp.depth
           best_center[0] =action.grasp.center[0]
           best_center[1] =action.grasp.center[1]


           best_angle = action.grasp.angle
           best_angle = float(best_angle)*180/3.141592
           
           print("gqcnn_best_center : \t"+str(x)+","+str(y))
           print("best_angle : \t"+str(best_angle))
           print("\n\n\n\n\n\n\nbest_Q_value : \t" +str(action.q_value))
           print("XYZ : \t" +str(x)+str(y)+str(z))
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

import matplotlib.patches as patches

import cv2
import numpy as np

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A

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
   
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            camera_fps=pyk4a.FPS.FPS_30,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510
    
    mask = np.ones((WIDTH,HEIGHT),dtype=np.uint8)*255
    while True:
        capture = k4a.get_capture()
        if np.any(capture.depth) and np.any(capture.color) :
            depth_img = np.array(capture.depth,dtype=np.uint16)
            color_img = np.array(capture.transformed_color,dtype=np.uint8)
            depth_img = depth_img/65535.0*100
            depth_im =DepthImage(depth_img.astype("float32"), frame=camera_intr.frame)
            color_im = ColorImage(np.zeros([HEIGHT,WIDTH ,3]).astype(np.uint8),
                          frame=camera_intr.frame)
            seg_mask =cv2.resize(mask.copy(),(WIDTH,HEIGHT),interpolation=cv2.INTER_CUBIC)
            segmask = BinaryImage(seg_mask)
           #print(depth_img.shape)

            normdepth_img = np.uint8(cv2.normalize(depth_img*65535/100, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)*255)
            backtorgb = cv2.cvtColor(normdepth_img,cv2.COLOR_GRAY2RGB)
            backtorgb=cv2.line(backtorgb,(int(best_center[0]),int(best_center[1])),(int(best_center[0]),int(best_center[1])+1),(255,0,0),3)
            cv2.imshow("k4a",backtorgb)
            cv2.imshow("k4a_color", color_img)
            
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break

    k4a.stop()

if __name__ == '__main__':
    t1 = threading.Thread(target=detect_img)
    t1.start()
    t2 = threading.Thread(target=run_gqcnn)
    t2.start()



