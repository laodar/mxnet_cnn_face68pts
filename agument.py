import cv2
import numpy as np
import shutil
import os

def write_lst(save_name,pts_float):
    global i_sample,f_lst
    pts_str = ''
    for i in range(68):
        pts_str += str(pts_float[i,0])+' '+str(pts_float[i,1])+' '
    f_lst.writelines(str(i_sample)+' '+pts_str+save_name+'\n')
    i_sample += 1


save_dir = 'ag_helen/'
try:
    os.mkdir(save_dir)
except:
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

f_lst_read = open('crop_helen.lst')
f_lst = open('ag_helen.lst','w')

pts_float_org = np.zeros([68,2])
n_ag = 10
img_sz = 137
target_sz = 128
i_sample = 0

for line in f_lst_read:

    tuples = line.split(' ')
    path = tuples[-1].strip()
    img_org = cv2.imread(path)

    for i in range(68):
        pts_float_org[i,0] = float(tuples[i*2+1])
        pts_float_org[i,1] = float(tuples[i*2+2])

    #pts_int = pts_float.astype(int)

    for i_ag in range(n_ag):

        pts_float = pts_float_org.copy()
        img = img_org.copy()

        is_flip = np.random.randint(0,2)
        if is_flip:
            pts_float[:,0] = img_sz - pts_float[:,0] - 1.0
            img = cv2.flip(img,1)

        rotate_angle = 30.0*np.random.rand(1)-15.0
        M = cv2.getRotationMatrix2D((img_sz/2.0,img_sz/2.0),rotate_angle,1.0)
        img = cv2.warpAffine(img,M,(img_sz,img_sz))
        pts_float = np.concatenate((pts_float,np.ones([68,1])),axis=1).dot(M.transpose())

        fx,fy = np.random.rand(2)*0.1+1.0
        img = cv2.resize(img,dsize=(0,0),fx=fx,fy=fy)
        rfx,rfy = float(img.shape[1]-1.0)/(img_sz-1.0),float(img.shape[0])/(img_sz-1.0)
        pts_float[:,0] = pts_float[:,0] * rfx
        pts_float[:,1] = pts_float[:,1] * rfy

        center_offset = np.array([img.shape[1] - img_sz, img.shape[0] - img_sz]) / 2.0
        crop_offset = np.random.randint(0, img_sz - target_sz + 1, 2) + center_offset.astype(int)
        left,top,right,bottom = crop_offset[0],crop_offset[1],\
                                crop_offset[0]+target_sz,crop_offset[1]+target_sz
        img = img[top:bottom,left:right,:]
        pts_float[:,0] -= left
        pts_float[:,1] -= top
        save_name = save_dir + str(i_ag)+ '_' + path.split('/')[-1].strip()
        cv2.imwrite(save_name, img)
        write_lst(save_name, pts_float)
        print i_sample, save_name,left,top,right,bottom,img.shape

f_lst.close()