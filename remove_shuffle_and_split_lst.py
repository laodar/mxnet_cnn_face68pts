
import numpy as np
import cv2
import shutil
import os


#error_paths = glob.glob('error_images/*.jpg')+glob.glob('error_images/*.png')
#error_names = [error_path.split('/')[-1] for error_path in error_paths]
#error_names = ['']

lst_from = ['ag_helen.lst','ag_lfpw.lst']
lst_to = ['train64.lst','val64.lst']
f_lst_to = [open(lst,'w') for lst in lst_to]
image_to = ['train64/','val64/']

for save_dir in image_to:
    try:
        os.mkdir(save_dir)
    except:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)

i_sample = [0,0]
lines = []

for lst in lst_from:
    f_lst_from = open(lst)
    for line in f_lst_from:
        lines.append(line)
    f_lst_from.close()

np.random.shuffle(lines)

for line in lines:
    tuples = line.split(' ')
    coords = [float(pts) for pts in tuples[1:-1]]
    path = tuples[-1].strip()
    img_name = path.split('/')[-1]
    if max(coords)<128.0 and min(coords) > 0.0:
        str_label = ''
        for i in range(len(coords)):
            str_label += ' '+str(coords[i])#/127.0*63.0)
        flag = int(np.random.rand(1)>0.8)
        path_to = image_to[flag]+img_name
        #cv2.imwrite(path_to,cv2.resize(cv2.imread(path),dsize=(64,64)))
        cv2.imwrite(path_to,cv2.imread(path))
        f_lst_to[flag].writelines(str(i_sample[flag])+str_label+' '+path_to+'\n')
        i_sample[flag] += 1
        print path_to

print i_sample[0],i_sample[1]

for f in f_lst_to:
    f.close()


