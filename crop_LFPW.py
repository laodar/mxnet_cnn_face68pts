import glob
import numpy as np
import cv2
import shutil
import os

img_dirs = ['/home/laodar/dataset/lfpw/trainset/','/home/laodar/dataset/lfpw/testset/']
img_format = '.png'
save_dir = 'crop_lfpw/'
f_lst = open('crop_lfpw.lst','w')

try:
    os.mkdir(save_dir)
except:
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

img_paths = []

def get_crop_box(pts,k):
    i_nose = 30
    center_x,center_y = pts[i_nose,0],pts[i_nose,1]
    min_x,min_y = np.min(pts[:,0]),np.min(pts[:,1])
    max_x,max_y = np.max(pts[:,0]),np.max(pts[:,1])
    r_left = abs(center_x-min_x)
    r_right = abs(center_x-max_x)
    r_top = abs(center_y-min_y)
    r_bottom = abs(center_y-max_y)
    r = max(r_left,r_right,r_top,r_bottom)
    R = k*r
    left = int(center_x - R)
    top = int(center_y - R)
    right = int(center_x + R)
    bottom = int(center_y + R)
    return left,top,right,bottom

def fix_pts_by_crop(pts,box):
    left, top, right, bottom = box
    pts[:,0] -= left
    pts[:,1] -= top
    return pts

def fix_pts_by_scale(pts,fx,fy):
    pts[:,0] *= fx
    pts[:,1] *= fy
    return pts

def write_lst(save_name,pts_float):
    global i_sample,f_lst
    pts_str = ''
    for i in range(68):
        pts_str += str(pts_float[i,0])+' '+str(pts_float[i,1])+' '
    f_lst.writelines(str(i_sample)+' '+pts_str+save_name+'\n')
    print i_sample,save_name
    i_sample += 1

target_sz = 137 #142-128=14
i_sample = 0

for img_dir in img_dirs:
    img_paths += glob.glob(img_dir+'*'+img_format)

for img_path in img_paths:

    pts_file = open(img_path[:-3]+'pts')

    img_org = cv2.imread(img_path)

    #print img.shape

    img_name = img_path.split('/')[-1]

    pts_float_org = np.zeros([68,2])

    for i in range(3):
        pts_file.next()

    for i in range(68):
        pts_str = pts_file.next().split(' ')
        pts_float_org[i,0] = float(pts_str[0])
        pts_float_org[i,1] = float(pts_str[1])

    for k in np.random.rand(4)*0.2+1.2:
        pts_float = pts_float_org.copy()
        img = img_org.copy()
        box = get_crop_box(pts_float,k)
        img_crop = img[box[1]:box[3],box[0]:box[2],:]
        img_shape = img_crop.shape
        if img_shape[0] > 1 and img_shape[1] > 1:
            sx = (target_sz - 1.0) / (img_shape[1]-1.0)
            sy = (target_sz - 1.0) / (img_shape[0]-1.0)
            pts_float = fix_pts_by_scale(fix_pts_by_crop(pts_float,box),sx,sy)
            save_name = save_dir + str(k) + '_' + img_name
            cv2.imwrite(save_name, cv2.resize(img_crop,dsize=(target_sz,target_sz)))
            write_lst(save_name,pts_float)

f_lst.close()