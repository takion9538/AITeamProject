from glob import glob
import shutil
import os

img_list = glob('plants/image/*.jpg')
label_list = glob('plants/label/*.txt')
print(len(img_list))
print(len(label_list))

from sklearn.model_selection import train_test_split

train_img_list, val_img_list, train_label_list, val_label_list = train_test_split(img_list, label_list, test_size=0.2, shuffle=True, random_state=1)
print(train_label_list[0])

for file in val_label_list :
    shutil.copy(file, 'plants/labels/valid')
print('done')