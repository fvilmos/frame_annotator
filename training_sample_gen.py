'''************************************************************************** 
Frame Annotator

frame annotator tool
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''
from datetime import datetime
import json
import uuid
import os
import glob
import numpy as np
import cv2
from utils import utils

# load config
jf = open(".\\utils\\ts_cfg.json",'r')
cfg_data=json.load(jf) 

data_dict = {}
count = 0
sdata=[]

data_gen_path = cfg_data["BASE_PATH"] + cfg_data["OUT_PATH"]
backgrounds = cfg_data["BASE_PATH"] + cfg_data["BACKGROUND_PATH"] + '*.*'
samples = cfg_data["BASE_PATH"] + cfg_data["SOURCE"] + "*.png"

background_list = glob.glob(backgrounds,recursive=True)
object_list = glob.glob(samples,recursive=True)
sdata = np.array(object_list)

print ("background img list:", len(background_list))
print ("sample img list:", len(object_list))

if (len(background_list)== 0) or (len(object_list) == 0):
    print ("Check input directories for data!", backgrounds, samples)
    exit(0)

# create folders to store data
def create_unique_folder():
    now = datetime.now()
    ts = int(now.timestamp()*10000)
    return str(ts) + "\\"

unique_folder = create_unique_folder()
out_path = data_gen_path + unique_folder

if os.path.exists(out_path) == False:
    os.makedirs(out_path)

f = open(out_path + "info.meta",'a')

# make dirctories
if os.path.exists(cfg_data["BASE_PATH"]) == False:
    os.makedirs(cfg_data["BASE_PATH"])

if os.path.exists(cfg_data["BASE_PATH"] + cfg_data["SOURCE"]) == False:
    os.makedirs(cfg_data["BASE_PATH"] + cfg_data["SOURCE"])

if os.path.exists(data_gen_path) == False:
    os.makedirs(data_gen_path)

if isinstance(cfg_data["OUT_IMG_SIZE"], list) == True:
    out_shape = (cfg_data["OUT_IMG_SIZE"][0], cfg_data["OUT_IMG_SIZE"][1])
else:
    out_shape = (1,1)


for s in sdata:
    # load image
    # takes the label name from the folder name that stors the iamges
    label = cfg_data["LABEL"]
    
    # read image with alpha channel, resize a overlay
    img_overlay = cv2.imread(s, -1)
    size_transform_percent = np.random.choice(np.array(cfg_data["SIZE_TRANSFORM_PERCENT"]),1)[0]
    img_overlay = cv2.resize(img_overlay, dsize=(int(img_overlay.shape[1]*size_transform_percent),int(img_overlay.shape[0]*size_transform_percent)), interpolation=cv2.INTER_LANCZOS4)    
    h,w,_ = img_overlay.shape

    # get random background
    bg_img_list = np.random.choice(background_list, size=cfg_data["NR_OF_BACKGROUNDS_TO_USE"])
    
    for bgi in bg_img_list:   
        id = str(uuid.uuid4().int)
        data_dict['id'] = id
        data_dict['source'] = s
        img_bgi = cv2.imread(bgi)
        
        # resize only if there is a list
        if isinstance(cfg_data["OUT_IMG_SIZE"], list) == True:
            img_bgi = cv2.resize(img_bgi,out_shape)
        bgi_shape = img_bgi.shape

        # randomyze only on integer type, else use a list 
        if isinstance(cfg_data["RANDOM_OBJECT_PLACEMENT"], int) == True:
            if isinstance(cfg_data["RANDOM_OBJECT_PLACEMENT"], int) == 1:
                # generate on random position
                try:
                    x = np.random.randint(0,bgi_shape[1]-w)
                    y = np.random.randint(0,bgi_shape[0]-h)
                except:
                    print ("Background size issue (too small)! Use OUT_IMG_SIZE, to reshape it first!")
                    exit(0)
            else:
                x = w//2
                y = h//2
        elif isinstance(cfg_data["OUT_IMG_SIZE"], list) == True:
            x = cfg_data["RANDOM_OBJECT_PLACEMENT"][0]
            y = cfg_data["RANDOM_OBJECT_PLACEMENT"][1]
        else:
            x = w//2
            y = h//2

        # constract the dict to be saved
        data_dict['orig_img_shape'] = img_bgi.shape
        data_dict['work_img_shape'] = img_bgi.shape
        data_dict['file'] = str(count) +"_"+ str(id) + str(cfg_data["OUT_FILE_EXTENSION"])
  
        # overlay object on the background
        size = np.random.uniform(low=1-size_transform_percent,high=1+size_transform_percent)
        img_bgi, bbox = utils.overly_img_with_alpha(img_bgi,img_overlay,(x,y))
        
        # constract the dict to be saved
        b_bbox = {}
        b_bbox['bbox'] = bbox
        b_bbox['label'] = str(label)
        b_bbox['confidence'] = "100"
        b_bbox['style'] = 'xyxy'
        data_dict['b_bbox'] = [b_bbox]
        
        d_dict_data = json.dumps(data_dict)
        
        # save to meta file
        f.write(d_dict_data)
        f.write('\n')
        cv2.imwrite(out_path + data_dict['file'],img_bgi)
        count +=1
f.close()
print ("Files and meta were generated in:", out_path)
