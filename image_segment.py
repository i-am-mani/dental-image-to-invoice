import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_all_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        rects.append(cv2.boundingRect(contour))
        
    return rects,contours

def draw_all_rects(rects, destination):
    output = destination.copy()
    idx = 1
    for rect in rects:
        [x, y, w, h] = rect
        cv2.rectangle(output, (x, y), (x + w, y + h), ((100 + idx*10) % 255, (100 + idx*10) % 255, 0), 2)
        idx+=1
    return output

def get_best_contours(image, top=10):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sortedCnts = sorted(contours, key=cv2.contourArea, reverse=True)
    areas = [(x, cv2.contourArea(x)) for x in contours]
    mean = np.percentile([area for (c, area) in areas], top)
    filteredAreas = filter(lambda x: x[1]>mean, areas)
    filteredContours = [c for (c, a) in filteredAreas]
    
    group = []
    for contour in filteredContours:
        [x, y, w, h] = cv2.boundingRect(contour)
        group.append(cv2.boundingRect(contour))
    return group

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def compute_hist(image):
    hist = cv2.calcHist([image], [0, 1,2], None, [8, 8, 8], [0, 256, 0, 256, 0, 255])
    hist = cv2.normalize(hist, hist).flatten()
    return hist



def remove_background(im_test,all_rects, base_hist):
    """ Filter bounding boxes which are similar to background by analysing histogram """
    background_filtered_rects = []
    removed_rects = []
    for rect in all_rects:
        (bx, by, bw, bh) = rect
        hist_img = im_test[by:by+bh, bx:bx+bw]
        bhist = compute_hist(hist_img.copy())
        base2b_hist = cv2.compareHist(base_hist, bhist,  cv2.HISTCMP_CORREL)

        if base2b_hist < 0.50:
            background_filtered_rects.append(rect)
        else:
            removed_rects.append(rect)
    return background_filtered_rects


    
def filter_rects(im_width:int, im_height:int, unfiltered_rects:list):
    '''Filter bounding boxes by their aspect_ratio and absolute width/heigh > 15% of total_width '''
    filtered_rect = []
    for rect in unfiltered_rects:
        (x,y,w,h) = rect
        if w > im_width//15 and h > im_height//15 and w/h > 0.5 and h/w > 0.5:
            filtered_rect.append(rect)
    return filtered_rect

def get_cordinates(rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    return (x1, x2, y1, y2)

def filter_nested_rects(im_test, rects):
    '''
        Filter nested rect by comparing normalized historgram - correaltional.
        
        Nested Bounding box will be eliminate if it outer bounding box has higher histogram score.
        Parent bounding box will be eliminate if it barely(20%) matches its nested.
        
        ----
        Solves issues such as: 
        1) Parent bounding box consisting of bounding boxes enclosing different products
        2) Filter nested bounding box which does not improve representation of enclosing product
        better then parent
    '''
    sorted_rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
    im_width,im_height = im_test.shape[1], im_test.shape[0]
    i, j = 0, len(sorted_rects) - 1
    w_ratio, h_ratio = im_width // 10, im_height // 10
    overlapping = []

    while i < (len(sorted_rects) - 1):
        (ax1,ax2, ay1, ay2) = get_cordinates(sorted_rects[i])
        (bx1, bx2, by1, by2) = get_cordinates(sorted_rects[j])

        (ax, ay, aw, ah) = sorted_rects[i]
        (bx, by, bw, bh) = sorted_rects[j]

        a_img = im_test[ay:ay+ah, ax:ax+aw]
        b_img = im_test[by:by+bh, bx:bx+bw]

        if (bx1 >= (ax1) and bx2 <= (ax2)) and (by1 >= (ay1) and by2 <= (ay2)):
            b_x = bx - ax
            b_y = by - ay

            a_sub_b = a_img.copy()
            a_sub_b[b_y:b_y+bh, b_x:b_x+bw] = [255,255,255]

            a_sub_b_hist = cv2.compareHist(compute_hist(a_sub_b.copy()), compute_hist(b_img.copy()), cv2.HISTCMP_CORREL)
            overlapping.append((a_sub_b, b_img, a_sub_b_hist))

            if a_sub_b_hist < 0.20:
                del sorted_rects[i]
                j = len(sorted_rects) - 1
                continue
            else:
                del sorted_rects[j]
                j = j - 1 if j-1>i and j>i else len(sorted_rects) - 1
                continue
        j -= 1

        if j == i:
            i += 1
            j = len(sorted_rects) - 1
    return sorted_rects


def merge_overlapping_rects(im_test, rects):
    ''' 
        Merge bounding boxes if they are similar to each other, compared by normalized color histogram.
        constrained by w_ratio and h_ratio which are 10% of w and h.
    '''
    sorted_rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
    sorted_rects = rects
    i = 0
    j = len(sorted_rects) - 1
    
    while i < (len(sorted_rects) - 1):
        (ax1,ax2, ay1, ay2) = get_cordinates(sorted_rects[i])
        (bx1, bx2, by1, by2) = get_cordinates(sorted_rects[j])

        (ax, ay, aw, ah) = sorted_rects[i]
        (bx, by, bw, bh) = sorted_rects[j]

        if ((bx1 > (ax1 - 5) and bx1 < (ax2 + 5) ) or (bx2 < (ax2 - 5) and bx2 > ax1)) and ((by1 > (ay1 - 5) and by1 < (ay2 + 5) ) or (by2 < (ay2 + 5) and by2 > (ay1 - 5))):
                a = im_test[ay:ay+ah, ax:ax+aw]
                b = im_test[by:by+bh, bx:bx+bw]
                ahist = compute_hist(a.copy())
                bhist = compute_hist(b.copy())
                
                a2b_hist = cv2.compareHist(ahist, bhist, cv2.HISTCMP_CORREL)
                
#                 show(a);show(b)
#                 print(a2b_hist)
                
                if(a2b_hist > 0.6):
                    n_x1,n_x2, n_y1, n_y2 = min(bx1, ax1), max(bx2, ax2), min(by1, ay1), max(by2, ay2)
                    sorted_rects[i] = (n_x1, n_y1, abs(n_x2-n_x1), abs(n_y2-n_y1))
                    del sorted_rects[j]
        if (j == i) or (j - 1 <= i):
            i += 1
            j = len(sorted_rects) - 1
        else:   
            j -= 1


    return sorted_rects


class ImageSegementation:
    
    def __init__(self, background_img_path):
        bg_img = cv2.imread(background_img_path)
        self.bg_hist = compute_hist(bg_img)
        
    def init_hed(self):
        PROTEXT_PATH = "./opencv-models/deploy.prototxt"
        MODEL_WEIGHTS = "./opencv-models/hed_pretrained_bsds.caffemodel"
        self.HED_DNN = cv2.dnn.readNet(PROTEXT_PATH, MODEL_WEIGHTS)    
        cv2.dnn_registerLayer('Crop', CropLayer)
    
    def segment(self, image_to_segment):
        print("tracing image")
        traced_edges = self._trace_edges(image_to_segment)
        print('image traced')
        im_canny = cv2.Canny(traced_edges, 100,255)
        all_rects, all_contours = get_all_contours(im_canny)
        
        im = image_to_segment.copy()
        im_width = image_to_segment.shape[1]
        im_height = image_to_segment.shape[0]
        
        print('filtering rects')
        filtered_rects = filter_rects(im_width, im_height, all_rects)
        filtered_rects = remove_background(im, filtered_rects, self.bg_hist)
        filtered_rects = filter_nested_rects(im, filtered_rects)
        filtered_rects = merge_overlapping_rects(im, filtered_rects)
        
        return filtered_rects

    
    def _trace_edges(self, im):
        inp = cv2.dnn.blobFromImage(im, scalefactor=1.0,swapRB=False,crop=False)
        self.HED_DNN.setInput(inp)
        output = self.HED_DNN.forward()
        output = output[0, 0] # since, shape will be 4 dimensional eg. (1, 1, 1560, 2080)
        output = 255 * output # the intensity are were normalized
        output = output.astype(np.uint8)
        return output

