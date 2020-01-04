
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

#--------------------gqcnn--------------------------------------------__#




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
import matplotlib.pyplot as plt
from ctypes import cdll
import threading
import time
import ctypes
import threading
from ctypes import cdll
import ctypes
from numpy.ctypeslib import ndpointer
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#lib = cdll.LoadLibrary('./SimpleViewer.so')
best_center=[0 ,0]
center=[0 ,0]
prev_q_value=0
q_value=0
prev_depth=0

i=0
depth_mem=None
first_flag=True
toggle = 1
t=0

INITIAL = 0
FIND_LOC= 1
MOVE_TARGET = 2
GRIP = 3
GRIP_BOX= 4
MOVE_HOME= 5
ALL_RESET = 6




np.random.seed(0)

image = np.ones((240,320),dtype=float)

state = INITIAL
no_find = 0
loc_count = 0
count =0

lib = cdll.LoadLibrary('./viewer_opengl.so')
st = lib.Foo_start
end = lib.Foo_end
dataread =lib.Foo_dataread
dataread_color =lib.Foo_dataread_color
dataread_depth =lib.Foo_dataread_depth
dataread_color_to_depth =lib.Foo_dataread_color_to_depth
dataread.restype = ndpointer(dtype=ctypes.c_uint8, shape=(720,1280,2))
dataread_color.restype = ndpointer(dtype=ctypes.c_uint8, shape=(720,1280,4))
dataread_depth.restype = ndpointer(dtype=ctypes.c_uint16, shape=(512,512))#ctypes.POINTE

dataread_color_to_depth.restype = ndpointer(dtype=ctypes.c_uint8, shape=(512,512,4))
convert_2d_3d = lib.Foo_convert_2d_3d
convert_2d_3d.restype = ndpointer(dtype=ctypes.c_float, shape=(3))#ctypes.POINTE

def run_gqcnn():
   global depth_im
   global color_im
   global segmask
   global policy
   global im3
   global ax3
   global prev_q_value
   global prev_depth
   global toggle
   global t
   global x
   global y
   global z
   global best_angle
   global depth_raw
   global im1
   global ax1
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

   best_angle = 0
   x=0.0
   y=0.5
   while 1:
     try:
       print("GQCNN IS RUNNING")
       #depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
       if "input_images" in policy_config["vis"] and policy_config["vis"][
            "input_images"]:
             im1.set_data(depth_im)
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
       except:
          no_valid_grasp_count = no_valid_grasp_count+1;
          time.sleep(0.3)
      # Vis final grasp.
       if policy_config["vis"]["final_grasp"]:
       # vis.imshow(segmask,
        #           vmin=policy_config["vis"]["vmin"],
       #           vmax=policy_config["vis"]["vmax"])
        #im3.set_data(rgbd_im.depth)

        center[0] = action.grasp.center[0]+96
        center[1] = action.grasp.center[1]+96
        q_value = action.q_value
        angle = float(action.grasp.angle)*180/3.141592
        print("center : \t"+str(action.grasp.center))
        print("angle : \t"+str(action.grasp.angle)) 
        print("Depth : \t"+str(action.grasp.depth)) 
        if(prev_q_value<action.q_value):

           #vis.grasp(action.grasp, scale=1, show_center=True, show_axis=True)
           #vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
           #    action.grasp.depth, action.q_value))
           prev_q_value = action.q_value
           prev_depth = action.grasp.depth
           best_center[0] =action.grasp.center[0]+96
           best_center[1] =action.grasp.center[1]+96
           convert_data=convert_2d_3d(int(best_center[0]),int(best_center[1]))
           x = -1*convert_data[0]/1000 +no_valid_move_y 
           y = (-1*convert_data[1]/1000)+0.41
           z = convert_data[2]/1000

          # x=-(best_center[0]-592)*0.00065625 +0.00#592 
          # y=-(best_center[1]-361)*0.000673611+0.46   #361 
           best_angle = action.grasp.angle
           best_angle = float(best_angle)*180/3.141592
           
           print("gqcnn_best_center : \t"+str(x)+","+str(y))
           print("best_angle : \t"+str(best_angle))
           print("Q_value : \t" +str(action.q_value))
        num_find_loc = num_find_loc+1

       # vis.show()
        #plt.show()
        time.sleep(0.3);
        #plt.close()
     except:

       time.sleep(1);
       pass

def imnormalize(xmax,image):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : image:image data.
    : return: Numpy array of normalize data
    """
    xmin = 0
    a = 0
    b = 255
    
    return ((np.array(image,dtype=np.float32) - xmin) * (b - a)) / (xmax - xmin)

def run_kinect():
    global image
    global depth_im
    global color_im
    global toggle
    global segmask
    global policy
    global image
    global im1
    global ax1
    global best_center
    global t
    global best_angle
    global prev_q_value
    global prev_depth
    global depth_img
    global depth_raw
    global state
    global no_find
    global z
    global center
    global q_value
    global angle
    global mean_sector
    global no_valid_grasp_count
    global color_to_depth_raw
    global grip_success
    global robot_collided
    global no_valid_move_y 

    global r_image
    global out_classes
    global out_boxes
    global r_class_names
    global out_scores
    if maskrcnn_run ==1:
       global img
       global result
       global color_img_raw


    print("_______________RUNKINECT_____________")
    cv2.namedWindow('DEPTH_REAL', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('DEPTH_REAL', 1000,1000)
    cv2.namedWindow('COLOR_TO_DEPTH', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('COLOR_TO_DEPTH', 1000,1000)
    cv2.namedWindow('SEGMENTATION', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SEGMENTATION', 1000,1000)
    prev_z = 0

    best_angle = 0
    class_color = {'apple':(0,0,255),'bae':(0,135,224),'banana':(0,255,255),'bread':(38, 201, 255),'cake':(157, 229, 252),'carrot':(0, 145, 255),'cucumber':(0, 184, 58),'gagi':(107, 0,98),'grape':(82, 0, 75),'green_apple':(0, 255, 179),'green_pepper':(0, 212, 149),'hamburger':(138, 218, 255),'kiwi':(85, 112, 95),'lemon':(0, 255, 251),'orange':(0, 200, 255),'peach':(217,230, 255),'pepper':(0,0,255),'pumkin':(
30, 100, 186),'tomato':(0,0,255)}
    max_image = 100
    while 1:

     try:
       print("KINECT IS ON1")
       depth_data  =np.array(dataread(),dtype=np.uint8)
       print("KINECT IS ON2")
       depth_real_data  =np.array(dataread_depth(),dtype=np.uint16)
       depth_real_img = depth_real_data[96:512-96,96:512-96]/5000;
       
      # if max_image < depth_real_data.max():
       #   max_image = depth_real_data.max()
      # print(max_image)

       color_to_depth_data = np.array(dataread_color_to_depth(),dtype=np.uint8)
       color_to_depth_img = color_to_depth_data[96:512-96,96:512-96,0:3];
       color_to_depth_raw  = color_to_depth_img.copy()
       if maskrcnn_run ==1:
           color_img_raw = color_to_depth_raw.copy()
       color_data  =np.array(dataread_color(),dtype=np.uint8)
       depth_img_ = depth_data.copy()/65535
       depth_raw = depth_data[:,:,0]
       depth_temp= depth_img_[:,:,0]
       depth_img = depth_temp.astype("float32")
       print("KINECT IS ON3")
       depth_img = (depth_raw)/255

       color_img = color_data[:,:,0:3]


       depth_im =DepthImage(depth_real_img.astype("float32"), frame=camera_intr.frame)
       color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)



       hsv = cv2.cvtColor(color_to_depth_img, cv2.COLOR_BGR2HSV)
       lower_red = np.array([50, 50, 50])
       upper_red = np.array([180, 180, 180])
       mask = cv2.inRange(color_to_depth_img, lower_red, upper_red)
    
       res = cv2.bitwise_and(color_to_depth_img,color_to_depth_img,mask= mask)

       print(best_center)
       print(best_angle)


       mask2 = np.mean(color_to_depth_img,axis=2)
       mask2[mask2>=50] = 255
       mask2[mask2<50] = 0
       mask2[mask2==255] = 1
       mask2[mask2==0] = 255
       mask2[mask2==1] = 0
       mask2 = np.uint8(mask2)

       mask[mask==255] = 1
       mask[mask==0] = 255
       mask[mask==1] = 0

       segmask_img  = depth_real_img.astype("float32")
       threshold = np.mean(segmask_img)*0.8
       #segmask_img[0:160,:] =  segmask_img[0:160,:]*1.1
       segmask_img[segmask_img<threshold] = 0
       segmask_img[segmask_img>threshold] = 255
       segmask_img[segmask_img==0] = 254
       segmask_img[segmask_img==255] = 0
       segmask_img[segmask_img==254] = 255

       #segmask_img[:130,:] = 0
       #segmask_img[257:,:] = 0
       #segmask_img[:,:37] = 0
       #segmask_img[:,282:] = 0
       segmask_img = np.uint8(segmask_img)

       r = 40

       res_mask =  cv2.bitwise_or(segmask_img,mask)
       res_mask = cv2.bitwise_or(res_mask,mask2)
       #res_mask[:52,:] = 0
       #res_mask[res_mask>100] = 255
       #res_mask[res_mask<=100] = 0
       
       #res_mask[0:73,:] = 0
       #res_mask[242:,:] = 0
       #res_mask[:,0:27] = 0
       #res_mask[:,282:] = 0

       segmask = BinaryImage(res_mask)
       depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_real_img*150, alpha=2), 1)
       #print("MEAN IMAGE")
       try:
          sector = color_to_depth_img.get()
          mean_sector = np.mean(sector[265:270,150:180])
          #print(mean_sector)
       except:
          sector = color_to_depth_img
          mean_sector = np.mean(sector[265:270,150:180])
          #print(mean_sector)
          pass
       #ddddddfdfdf= np.array(color_img_raw.get())
      # mean_img = np.array(color_img_raw[256:276,143:180],dtype=float)
       try:
         r=80
         color_to_depth_img = cv2.line(color_to_depth_img, (int(best_center[0]-96-r*math.cos(best_angle/180*math.pi)),int(best_center[1]-96-r*math.sin(best_angle/180*math.pi))), (int(best_center[0]-96+r*math.cos(best_angle/180*math.pi)),int(best_center[1]-96+r*math.sin(best_angle/180*math.pi))), (0,0,255), 2)
         color_to_depth_img = cv2.line(color_to_depth_img, (int(best_center[0]-96),int(best_center[1]-96)), (int(best_center[0]-96+1),int(best_center[1]-96)) , (0,136,255), 3)
       except:
       	 pass
       cv2.imshow('COLOR_TO_DEPTH',color_to_depth_img )
       cv2.imshow('DEPTH_REAL',np.uint8( depth_colormap ))
       cv2.imshow('SEGMENTATION',np.uint8( res_mask ))
       cv2.waitKey(1)





       time.sleep(0.01)
     except:
       print("KINECT_EXEPTION")
       time.sleep(0.01)
 
if __name__ == '__main__':
    global depth_im
    global color_im
    global segmask
    global policy
    global im1
    global ax1
    global x
    global y
    global best_angle
    global depth_img
    global num_find_loc
    global grip_time
    global grip_success

    model_name = "GQCNN-4.0-PJ"
    depth_im_filename = "dframe1.npy"
    segmask_filename = "segmask1.jpg"
    camera_intr_filename = "kinect.intr"
    config_filename = None
    fully_conv = 0
    maskrcnn_run = 0
    
    # Set model if provided.
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"gqcnn/models")
    model_path = os.path.join(model_dir, model_name)

    # Get configs.
    model_config = json.load(open(os.path.join(model_path, "config.json"),
                                  "r"))

    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        if input_data_mode == "tf_image":
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == "tf_image_suction":
            gripper_mode = GripperMode.LEGACY_SUCTION
        elif input_data_mode == "suction":
            gripper_mode = GripperMode.SUCTION
        elif input_data_mode == "multi_suction":
            gripper_mode = GripperMode.MULTI_SUCTION
        elif input_data_mode == "parallel_jaw":
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError(
                "Input data mode {} not supported!".format(input_data_mode))
    # Set config.
    if config_filename is None:
        if (gripper_mode == GripperMode.LEGACY_PARALLEL_JAW
                or gripper_mode == GripperMode.PARALLEL_JAW):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "/home/sung/gqcnn/cfg/examples/fc_gqcnn_pj.yaml")
            else:
                config_filename = os.path.join(
                    "gqcnn_pj_kinect.yaml")
        elif (gripper_mode == GripperMode.LEGACY_SUCTION
              or gripper_mode == GripperMode.SUCTION):
            if fully_conv:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "/home/sung/gqcnn/cfg/examples/fc_gqcnn_suction.yaml")
            else:
                config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    "/home/sung/gqcnn/cfg/examples/gqcnn_suction.yaml")


   # Read config.
    config = YamlConfig(config_filename)
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]

    # Make relative paths absolute.
    if "gqcnn_model" in policy_config["metric"]:
        policy_config["metric"]["gqcnn_model"] = model_path
        if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
            policy_config["metric"]["gqcnn_model"] = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                policy_config["metric"]["gqcnn_model"])

    # Setup sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.
    depth_data =np.load(depth_im_filename)
    #depth_data = np.array((depth_data-depth_data.min())/(depth_data.max()-depth_data.min()),dtype=float)
    depth_data = np.array((depth_data[:,:,0]-depth_data[:,:,0].min())/(depth_data[:,:,0].max()-depth_data[:,:,0].min()),dtype=float)
    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                    3]).astype(np.uint8),
                          frame=camera_intr.frame)

    # Optionally read a segmask.
    segmask = None
    if segmask_filename is not None:
        segmask = BinaryImage.open(segmask_filename)
    valid_px_mask = depth_im.invalid_pixel_mask().inverse()
    if segmask is None:
        segmask = valid_px_mask
    else:
        segmask = segmask.mask_binary(valid_px_mask)
   # print("----------------------------------")


     # Set input sizes for fully-convolutional policy.
    if fully_conv:
         policy_config["metric"]["fully_conv_gqcnn_config"][
             "im_height"] = depth_im.shape[0]
         policy_config["metric"]["fully_conv_gqcnn_config"][
             "im_width"] = depth_im.shape[1]

     # Init policy.
    if fully_conv:
         # TODO(vsatish): We should really be doing this in some factory policy.
         if policy_config["type"] == "fully_conv_suction":
             policy = FullyConvolutionalGraspingPolicySuction(policy_config)
         elif policy_config["type"] == "fully_conv_pj":
             policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
         else:
             raise ValueError(
                "Invalid fully-convolutional policy type: {}".format(
                    policy_config["type"]))
    else:
         policy_type = "cem"
         if "type" in policy_config:
            policy_type = policy_config["type"]
         if policy_type == "ranking":
            policy = RobustGraspingPolicy(policy_config)
         elif policy_type == "cem":
            policy = CrossEntropyRobustGraspingPolicy(policy_config)
         else:
            raise ValueError("Invalid policy type: {}".format(policy_type))

    t0 = threading.Thread(target=st)
    t0.start()
    t1 = threading.Thread(target=run_kinect)
    t1.start()
    t3 = threading.Thread(target=run_gqcnn)
    t3.start()

    t4 = threading.Thread(target=move_indy)
    t4.start()

    if maskrcnn_run == 1 :
      t5 = threading.Thread(target=run_maskrcnn)
      t5.start()
    close_motor()
    #ax1 = plt.subplot(111)
    temp = np.ones([240,320,3])*255
    #im1 = ax1.imshow(temp)
    #plt.ion()
  

    plt.ion()

    global color_to_depth_raw
    global r_image
    global out_classes
    global out_boxes
    global r_class_names
    global out_scores
    color_to_depth_raw = np.zeros((320,320),dtype=np.uint8)

    time.sleep(0.01) 
    plt.ioff()
    end()
