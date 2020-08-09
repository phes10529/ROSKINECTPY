#!/usr/bin/env python3
#!coding=utf-8
#modified by leo at 2018.04.26

import rospy
import cv2
import sys
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
from ctypes import *
import math
from PIL import Image as Image2

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("/home/nano/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image.encode('utf-8'), 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

# 2018.04.25
def showPicResult(image):
    img = cv2.imread(image)
    global out_img
    cv2.imwrite(out_img, img)
    for i in range(len(r)):
        x1=r[i][2][0]-r[i][2][2]/2
        y1=r[i][2][1]-r[i][2][3]/2
        x2=r[i][2][0]+r[i][2][2]/2
        y2=r[i][2][1]+r[i][2][3]/2
        im = cv2.imread(out_img)
        cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
        #putText
        x3 = int(x1+5)
        y3 = int(y1-10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if ((x3<=im.shape[0]) and (y3>=0)):
            im2 = cv2.putText(im, str(r[i][0]), (x3,y3), font, 1, (0,255,0) , 2)
        else:
            im2 = cv2.putText(im, str(r[i][0]), (int(x1),int(y1+6)), font, 1, (0,255,0) , 2)
        #This is a method that works well. 
        cv2.imwrite(out_img, im)
    cv2.imshow('yolo_image_detector', cv2.imread(out_img))
    # can't be '0', or can't make callback loop.
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def callback(data):
    global count
    count = count + 1

    # pass some frames, detect the last frame.
    if count == 5:
        count = 0
        # This place con't be data.data, or it will be str_type.
        global bridge
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
        print ('*** Have subscribed webcam_frame. ***')
        img_arr = Image2.fromarray(cv_img)
        global video_tmp
        img_goal = img_arr.save(video_tmp)
        global r,net,meta,yolo_class_pub, yolo_roi_pub
        r = detect(net, meta, video_tmp)
        #print r
        for j in range(len(r)):
            print(r[j][0], ' : ', int(100*r[j][1]),"%")
            print(r[j][2])
        #print ''
        #print '#-----------------------------------#'

        try:
            for i in range(len(r)):
                yolo_class_str = '** Get the object_pose in this image. ***' +' : '+ str(r[i][0])
                yolo_class_pub.publish(yolo_class_str)
                print('** Get the object_pose in this image. ***' +' : '+ str(r[i][0]))
                x1=r[i][2][0]-r[i][2][2]/2
                y1=r[i][2][1]-r[i][2][3]/2
                ROI = RegionOfInterest()
                ROI.x_offset = int(x1)
                ROI.y_offset = int(y1)
                ROI.width = int(r[i][2][2]/2)
                ROI.height = int(r[i][2][3]/2)
                # ROI.do_rectify = False
                yolo_roi_pub.publish(ROI)
                print(('** Publishing Yolo_ROI succeed. --%d ***', i+1))
        except:
            rospy.loginfo("Publishing ROI failed")
        print('')

        #display the rectangle of the objects in window
       # showPicResult(yolo_image_detector)
        
    else:
        pass

def yoloDetect():
    #init ros_node
    rospy.init_node('yolo_detetor', anonymous=True)
    #load config_file
    global net
    net = load_net("/home/nano/darknet/cfg/yolov3-tiny.cfg".encode("utf-8"), "/home/nano/darknet/weights/yolov3-tiny.weights".encode("utf-8"), 0)
    global meta
    meta = load_meta("/home/nano/darknet/cfg/coco.data".encode("utf-8"))
    
    global out_img
    out_img = "/home/nano/darknet/test_result.jpg"
    global video_tmp
    video_tmp = "/home/nano/darknet/video_tmp.jpg"

    global bridge
    bridge = CvBridge()
    #subscribe and publish related topic
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback)
    global yolo_class_pub, yolo_roi_pub
    yolo_class_pub = rospy.Publisher('/yolo_class', String, queue_size=10)
    yolo_roi_pub = rospy.Publisher("/yolo_roi", RegionOfInterest, queue_size=1)
    global count,r
    count = 0
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    yoloDetect()
