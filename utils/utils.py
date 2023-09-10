'''************************************************************************** 
Frame Annotator

frame annotator tool
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''
import numpy as np
import cv2
import json
import os
import glob


def load_data(file_name):
    '''
    Load meta data from recorded files (meta)
    '''
    f = open(file_name, "r")
    lines = f.readlines()
    data = []
    
    tmp = str.split(file_name,"\\")
    # reconstruct file path
    dir_name =''
    for p in range(len(tmp)-1):
        dir_name +=tmp[p] + '\\'

    # read line by line, de-serialize to dict
    for l in lines:
        try:
            obj_dict = json.loads(l)
        except:
            print("exception:", l)
            exit(0)
        
        # test if img exist
        test_img_path = os.path.exists(dir_name + obj_dict['file'])
        if test_img_path == True:
            file = obj_dict['file']
            obj_dict['file'] = dir_name+file
            # get all objects, split as many objects are detected
            for det in obj_dict['b_bbox']:                
                # copy the original dict, add a single item to the list of 'b_bbox'
                l_obj = obj_dict.copy()
                l_obj['b_bbox'] = [det]
                
                data.append(l_obj)
        
    f.close()
    return np.array(data)


def sam_segment(img_in,ret,alpha_ch=0):
    """
    generate a mask overlayed with hte original image
    Args:
        img_in (_type_): input image
        ret (_type_): segmentation masks
        alpha_ch (int, optional): use alpha channel Defaults to 0.

    Returns:
        _type_: masked rgb image
    """
    masks_arr = []
    if ret[0] is not None:
        masks = ret[0]  

        for mask in masks:
            # Get the first mask
            mask = np.uint8(mask)

            # Apply the mask to the image
            segmented_img = cv2.bitwise_and(img_in, img_in, mask=mask)
            if alpha_ch == 1:
                segmented_img = np.dstack((segmented_img,cv2.bitwise_not(mask)))
            masks_arr.append(segmented_img)
        segmented_img = np.stack(masks_arr)
        return segmented_img

def img_format(img,bbox=None,out_size=(256,256)):
    """
    Format image, takeing the longest dimention (w,h)
    Args:
        img (_type_): input image RGB
        bbox (_type_, optional): bounding box [x1,y1,x2,y2]. Defaults to None.
        out_size (tuple, optional): out out img dimension. Defaults to (256,256).

    Returns:
        _type_: new image, adjusted bbox
    """
    ih,iw,ch = img.shape
    max_size = max(ih,iw)
    ratio = max_size / out_size[0]
    h_n = int(ih/ratio)
    w_n = int(iw/ratio)

    img_n = cv2.resize(img, (w_n,h_n), interpolation=cv2.INTER_LANCZOS4)
    img_res = np.zeros([out_size[0],out_size[1],ch], dtype=np.uint8)
    
    img_res[:h_n,:w_n] = img_n

    bbox_n = None
    if bbox is not None:
        x,y,w,h = bbox
        bbox_n = [int((x-0.5*w)*iw/ratio), int((y-0.5*h)*ih/ratio), int(w*iw/ratio), int(h*ih/ratio)]
    
    return img_res, bbox_n

def puttext_bg(img,text='',position=(10,160), font_type=None, font_size=0.4, font_color=[0,255,0],bg_color=[0,0,0],font_thickness=1):
    """
    Text with background color
    Args:
        img: input image
        text (str, optional): Defaults to ''.
        position (tuple, optional): Defaults to (10,160).
        font_type (_type_, optional): Defaults to None.
        font_size (float, optional): Defaults to 0.4.
        font_color (list, optional): Defaults to [0,255,0].
        bg_color (list, optional): Defaults to [0,0,0].
        font_thickness (int, optional): Defaults to 1.
    """

    if font_type is None:
        font_type = cv2.FONT_HERSHEY_SIMPLEX
    (t_w,t_h),_ = cv2.getTextSize(text, font_type, font_size, font_thickness)
    cv2.rectangle(img, position, (position[0] + t_w, position[1] + t_h), bg_color, -1)
    cv2.putText(img,text ,(position[0], int(position[1]+t_h+font_size-1)),font_type,font_size,font_color,font_thickness)

def plot_bbox(img, label="", bbox=[10,10,30,30], color=[0,255,0], shape=(256,256), factor=1, style="xywh"):
    """
    Plot bounding box 

    Args:
        img (_type_): input RGB image
        label (str, optional): class name. Defaults to "".
        bbox (list, optional): bounding box. Defaults to [10,10,30,30].
        color (list, optional): box color. Defaults to [0,255,0].
        shape (tuple, optional): image shape (h,w). Defaults to (256,256).
        factor (int, optional): multiplication factor. Defaults to 1.
        style (str, optional): box stype xyxy or xywh. Defaults to "xywh".

    Returns:
        _type_: image with owerlayed bbox
    """
    img = cv2.resize(img,shape)
    bbox *= factor
    x0,y0,x1,y1 = bbox
    
    if style=="xywh":
        cv2.rectangle(img,(int(x0-x1//2),int(y0-y1//2)),(int(x0+x1//2),int(y0+y1//2)),color,2)
    else:
        cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),color,2)
    puttext_bg(img,text=str(label),position=(int(x0+5),int(y0-5)),bg_color=[0,0,0])
    return img

def rotate_img(img,angle,rot_center=None,scale=1.0):
    """
    Simple rotates an image
    Args:
        img (_type_): input image
        angle (_type_): angle
        rot_center (_type_, optional): usually w//2, h//2. Defaults to None.
        scale (float, optional): magnification factor. Defaults to 1.0.

    Returns:
        _type_: rotated image
    """
    h,w,_ = img.shape
    if rot_center is None:
        rot_center = (w/2,h/2)
    rm = cv2.getRotationMatrix2D(rot_center, int(angle), scale)
    res = cv2.warpAffine(img, rm, (w,h), flags=cv2.INTER_LINEAR)
    return res

class DetectInputSource():
    """
    Detects the type of the input source, and abstratatize for usage
    """
    def __init__(self, source, cfg_data=None) -> None:
        self.source = source
        self.file_name = str(source)
        self.isImg = False
        self.isMeta = False
        self.isVideo = False
        self.process_file = None
        self.count = 0
        self.sdata = []
        self.cfg_data = cfg_data

        # detect input types
        if isinstance(source, str)==True:
            print ("search path:", self.cfg_data["BASE_PATH"]+self.cfg_data["SOURCE"])
            self.process_file = glob.glob(self.cfg_data["BASE_PATH"]+self.cfg_data["SOURCE"], recursive=True)
            if len(self.process_file) == 0: 
                print ("No file to process, check config SOURCE value!!!")
                exit(0)

            self.file_name  = self.process_file [self.cfg_data["FILE_INDEX"]]
            print ("SOURCE LIST:", self.process_file)
            print ("SOURCE:", self.cfg_data["BASE_PATH"]+self.cfg_data["SOURCE"])
            
            # test if input is an image
            if sum([1 if a in source else 0 for a in self.cfg_data["ACCEPTED_IMAGE_TYPES"]]) > 0:
                self.cap = 1
                self.isImg  = True
            # maybe meta
            elif ".meta" in self.cfg_data["SOURCE"]:
                self.cap = 1
                self.isMeta = 1
                print (self.process_file)
                for s in self.process_file:
                    print ("Loading:",s)
                    self.sdata.extend(load_data(s))
            # or mp4
            elif ".mp4" in self.cfg_data["SOURCE"]:
                self.isVideo = True
                self.cap = cv2.VideoCapture(self.file_name)
            # assume is some video file
            else:
                self.cap = cv2.VideoCapture(self.file_name)
        else:
            print ( "SOURCE:", int(self.cfg_data["SOURCE"]))
            self.cap = cv2.VideoCapture(int(self.cfg_data["SOURCE"]))
    
    def device_opened(self):
        """
        Abstraction for device

        Returns:
            _type_: [True/False] dependant of device status
        """
        ret = False
        if isinstance(self.source, int):
            ret = self.cap.isOpened()
        return ret
    
    def read(self, count=None):
        """
        Read any type of data (abstraction)
        Returns:
            _type_: image is returned
        """
        if count is not None: 
            self.count = count
        img = None

        # web cam
        if isinstance(self.source, int): 
            _,img = self.cap.read()
        # image
        elif self.isImg == True:
            if self.count < len(self.process_file):
                self.file_name = self.process_file[self.count]
                img = cv2.imread(self.file_name)
                if count is None:
                    self.count +=1
            else:
                exit(0)
        # meta file
        elif self.isMeta == True:
            if self.count < len(self.sdata):
                file_name = self.sdata[self.count]["file"]
                print (file_name)
                img = cv2.imread(file_name)
                if count is None:
                    self.count +=1
            else:
                exit(0)
        # video
        elif self.isVideo == True:
            _,img = self.cap.read()
        else:
            print ("Unsupported extension!")

        return img

def get_bbox_from_mask(mask):
    """
    create a mask from the segmented image
    Args:
        img_w (_type_): work image
        mask (_type_): segmented mask
    Returns:
        x1,y1,x2,y2 - bounding box coordinates
    """
    # get last channel, detect the countour, get bbox
    l_mask = cv2.bitwise_not((np.array(mask[...,-1])-254)*255)
    contours, _ = cv2.findContours(l_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # get biggerst ignore the rest, generate the bbox
    countour_max = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(countour_max)
    
    return x,y,x+w,y+h

def overly_img_with_alpha(img_bg, img_alpha, pos=(0,0),scale=0):
    '''
    img_bg - background image (bgr format)
    img_alpha - image to overlay shape=(h,w,4) (bgr format, last channel is the alpha channel)
    pos - position to render the image to overlay
    scale - scale of the overlayed image, 0% no resize
    
    '''
    if img_alpha.shape[-1] < 4:
        print ("no alpha channel for: img_alpha!")
        exit(0)

    img_bg_l = img_bg.copy()
    img_alpha_r = img_alpha.copy()

    # resize image, if needed
    if scale != 0:
        n_w = int(img_alpha_r.shape[1] * scale / 100)
        n_h = int(img_alpha_r.shape[0] * scale / 100)
        img_alpha_r = cv2.resize(img_alpha_r, (n_w,n_h), interpolation = cv2.INTER_LANCZOS4)

    # create mask and bgr image
    b,g,r,i_mask = cv2.split(img_alpha_r)
    img_bgr = cv2.merge((b,g,r))
    
    # normalize mask to 0->255
    i_mask = (i_mask-254)*255
    l_h,l_w = i_mask.shape

    # format bbox coordinates
    x1 = pos[0]
    y1 = pos[1]
    x2 = x1 + l_w
    y2 = y1 + l_h

    # make roi from the background
    roi = img_bg_l[y1:y2,x1:x2]

    # create masks with the original image parts
    img_roi_bg = cv2.bitwise_and(np.uint8(roi),np.uint8(roi),mask = np.uint8(i_mask))
    img_roi_fg = cv2.bitwise_and(img_bgr,img_bgr,mask = cv2.bitwise_not(np.uint8(i_mask)))
    img_bg_l[y1:y2, x1:x2] = cv2.add(img_roi_bg,img_roi_fg)

    return img_bg_l,[x1,y1,x2,y2]