'''************************************************************************** 
Frame Annotator

frame annotator tool
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''
import numpy as np
import cv2
import json
import torch
from utils import utils
from mobile_sam import sam_model_registry, SamPredictor
import uuid
import os

# variables
click_list = []
count = 0
loaded = False
last_click_len = 0
info = {"1":"c - clear mouse points", "2":"n - next frame", "3":"s - save", "4":"ESC - escape"}

# load config
jf = open(".\\utils\\fa_cfg.json",'r')
cfg_data=json.load(jf) 

out_path = cfg_data["BASE_PATH"] + cfg_data["OUT_PATH"]

# get device to run mobile SAM
if cfg_data["FORCE_CPU"] == 1:
    torch.cuda.is_available = lambda : False

if os.path.exists(out_path) == False:
    os.makedirs(out_path)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SAM weiths
path = cfg_data["MODEL_PATH_MSAM"]
model = sam_model_registry["vit_t"](checkpoint=path)

# assign to device
model.to(device=DEVICE)
model.eval()
predictor = SamPredictor(model)

# mouse callback
def get_mouse_event(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDBLCLK:
     click_list.append([x,y])

# assigne the mouse event to window
cv2.namedWindow('Data')
cv2.setMouseCallback('Data',get_mouse_event)

# input type abstractor
device_test = utils.DetectInputSource(cfg_data["SOURCE"], cfg_data=cfg_data)

while(device_test):

    k = cv2.waitKey(10)
    # take next file
    if k == ord('n'):
        count+=1
        last_click_len = -1
        loaded = False

    # load / reload file if needed
    if loaded == False:
        img = device_test.read(count)
        img_out = np.zeros(shape=[0,0,4])

        if isinstance(cfg_data["WORK_IMG_SIZE_HW"], list):
            img = cv2.resize(img,(cfg_data["WORK_IMG_SIZE_HW"][1], cfg_data["WORK_IMG_SIZE_HW"][0]))
        else:
            h,w,_ = img.shape
            img = cv2.resize(img,(w//int(cfg_data["WORK_IMG_SIZE_HW"]), h//int(cfg_data["WORK_IMG_SIZE_HW"])))
            
        img_work = img.copy()
        loaded = True
    
    utils.puttext_bg(img_work,str(count),(10,10))

    # clear all clicks
    if k == ord('c'):
        click_list=[]

    # segment if new point available
    if len(click_list) != last_click_len:
        last_click_len = len(click_list)
        
        if len(click_list) == 0 :
            print ("no ponts to segment! use the mouse double - click")
            loaded = False
            continue

        #segment
        predictor.set_image(img)

        input_point = np.array(click_list)
        input_label = np.array([1]*len(click_list))

        ret = predictor.predict(point_coords=input_point, point_labels=input_label, box=None,multimask_output=False)

        # get segmented images
        img_roi_n = utils.sam_segment(img,ret, alpha_ch=1)
        
        # show segmented, blend the rest    
        img_work = cv2.addWeighted(img, 0.3, img_roi_n[0][...,:3], 0.7, 0)
        
        x1,y1,x2,y2 = utils.get_bbox_from_mask(img_roi_n[0])
        cv2.rectangle(img_work,(x1,y1),(x2,y2),(0,0,255),2)
        
        if cfg_data["WRITE_TYPE"] == 1:
            img_masked_roi = img_roi_n[0][y1:y2,x1:x2]
            img_out = img_masked_roi
            cv2.imshow("Segmented",img_masked_roi)

        # show segmented iamge
        if cfg_data["WRITE_TYPE"] == 2:
            img_out = img_roi_n[0]
            cv2.imshow("Segmented",img_roi_n[0])

    # show mouse clicks
    for p in click_list:
        cv2.circle(img_work,(int(p[0]), int(p[1])), 3,[0,255,0], -1)
    
    #display the info
    i = 0
    for ke,va in info.items():
        utils.puttext_bg(img_work,str(va),(10,10+i))
        i +=10

    # save file in the out directory
    if k == ord("s"):
        out_path = cfg_data["BASE_PATH"] + cfg_data["OUT_PATH"]
        f_name = str(count) + "_" + str(uuid.uuid4().int) + ".png"

        print ("writing:", f_name)
        cv2.imwrite(out_path + f_name,img_out)

    # exit on escape
    if k == 27:
        exit(0)

    # show picture
    cv2.imshow("Data", img_work)
cv2.destroyAllWindows()