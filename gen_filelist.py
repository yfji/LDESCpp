import os
import os.path as op

sequence='Mhyang'
cur_dir=os.getcwd()
img_dir=op.join(cur_dir, 'sequences\\{}\\img'.format(sequence))
print(img_dir)

img_files=os.listdir(img_dir)
img_files=[f for f in img_files if f[-3:]=='jpg']

img_list=open('./{}.txt'.format(sequence),'w')
label_file=open('./{}_label.txt'.format(sequence),'w')

gt_file=open(op.join(cur_dir, 'sequences/{}/groundtruth_rect.txt'.format(sequence)), 'r')
lines=gt_file.readlines()

for ix, img_file in enumerate(img_files):
    full_path=op.join(img_dir, img_file)
    img_list.write(full_path+'\n')
    rect=lines[ix].rstrip()
    if ',' in rect:
        xyxy=rect.split(',')
    else:
        xyxy=rect.split()
    
    newline=','.join(xyxy)

    label_file.write(newline+'\n')

img_list.close()
label_file.close()
gt_file.close()
print('Done')
    