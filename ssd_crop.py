import shapely.geometry as shgeo
import pdb
import random
import cv2
import numpy as np
def iou_polygon(poly1,poly2):
    try:
        poly1=shgeo.Polygon(poly1)
        poly2=shgeo.Polygon(poly2)
        intersection=poly2.intersection(poly1)
        if poly1.area==0 or poly2.area==0:
            return 0
        return intersection.area/(poly1.area+poly2.area-intersection.area)
    except:
        return 0

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def min_dis(out_poly):
    dis=[]
    for i in range(4):
        
        dis.append(np.linalg.norm(out_poly[(i+1)%4]-out_poly[i]))
    return min(dis)

def jaccard_numpy(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / area_a  # [A,B]


def crop_ssd(image,polygon,h_crop):
    height, width, _ = image.shape
    polygon=polygon.reshape(-1,4,2)
    boxes=np.empty([len(polygon),4],dtype=np.float32)
    boxes[:,0]=np.min(polygon[:,:,0],axis=1)
    boxes[:,1]=np.min(polygon[:,:,1],axis=1)
    boxes[:,0+2]=np.max(polygon[:,:,0],axis=1)
    boxes[:,1+2]=np.max(polygon[:,:,1],axis=1)
    while True:
        # randomly choose a mode
        mode = (0.1,None)

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        # max trails (50)
        for _ in range(50):
            current_image = image


            w = random.uniform(0.1 * width, width)
            h = random.uniform(0.1 * height, height)
            
            # aspect ratio constraint b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            # aspect ratio constraint b/t .5 & 2
            #if h / w < 0.5 or h / w > 2:
            #    continue

            left = random.uniform(0,width - w)
            top = random.uniform(0,height - h)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])
            if (rect[::2]<0).any() or (rect[::2]>width).any() or (rect[1::2]<0).any() or (rect[1::2]>height).any():
                pdb.set_trace()

            # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
            overlap = jaccard_numpy(boxes, rect)

            # is min and max overlap constraint satisfied? if not try again
            if overlap.max() <0.1:
                continue
            
            return (rect[0],rect[1],rect[2],rect[3])
        print('crop_false')
        return (0,0,width,height)

def crop_label(polys,crop_hw):
    radom_h,end_h,random_w,end_w=crop_hw
    left=random_w
    up=radom_h
    right=end_w
    down=end_h
    thres=0.2
    min_h=10
    imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])

    instance=shgeo.Polygon(polys.reshape(-1,2))
    poly1=instance
    poly2=imgpoly
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    poly1_area = poly1.area
    if poly1_area==0:
        print('wrong')
        pdb.set_trace()
    half_iou = inter_area / poly1_area

    if half_iou==1:
        
        polys[::2]-=left
        polys[1::2]-=up
        if min_dis(polys.reshape(-1,2))<10:
            return 2,polys.reshape(-1)

        return 1,polys.reshape(-1)
    if half_iou>0:
        inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
        out_poly = list(inter_poly.exterior.coords)[0: -1]

        out_poly=np.array(out_poly,dtype=np.float32).reshape(-1,2)

        out_poly[:,0]-=left
        out_poly[:,1]-=up

        if len(out_poly)==5:
            
            out_poly_before=out_poly
            out_poly=cv2.boxPoints(cv2.minAreaRect(out_poly.reshape(-1,2))).astype(np.float32,copy=False).reshape(-1,2)
            if iou_polygon(out_poly,out_poly_before)<0.7:

                return 2,out_poly.reshape(-1)
        if len(out_poly)==4 and half_iou>thres:
            polys_2=polys.reshape(-1,2)
            if min_dis(out_poly)/(min_dis(polys_2)+1e-10)<0.6 or min_dis(out_poly)<10:
                return 2,out_poly.reshape(-1)
            return 1,out_poly.reshape(-1)
        if len(out_poly)==4 and half_iou<=thres:
            return 2,out_poly.reshape(-1)


        if len(out_poly) >=3:
            out_poly=cv2.boxPoints(cv2.minAreaRect(out_poly.reshape(-1,2))).astype(np.float32,copy=False).reshape(-1)
            return 2,out_poly.reshape(-1)

    return 3,None





def ssd_crop(im,polyinters,tags,h_crop):

    polyinters=polyinters.reshape(-1,8)

    crop_points=crop_ssd(im,polyinters,h_crop)
    rect=crop_points
    im = im[rect[1]:rect[3], rect[0]:rect[2]]
    h,w=im.shape[:2]

    im=cv2.resize(im,(h_crop,h_crop))  
    w_scale=float(h_crop)/w
    h_scale=float(h_crop)/h
    polyinters[:,::2]*=w_scale
    polyinters[:,1::2]*=h_scale
    crop_points=(rect[1]*h_scale,rect[3]*h_scale, rect[0]*w_scale,rect[2]*w_scale)


    is_valid=np.ones(len(polyinters),dtype=np.bool)


    for i_roi in range(len(is_valid)):
        flag,out_poly=crop_label(polyinters[i_roi],crop_points)
        

        if flag ==3:
            is_valid[i_roi]=False
        elif flag==1:
            polyinters[i_roi]=out_poly
        elif flag==2:
            polyinters[i_roi]=out_poly
            tags[i_roi]=1
    polyinters=polyinters[is_valid]
    tags=tags[is_valid]

    return im,polyinters.reshape(-1,4,2),tags
