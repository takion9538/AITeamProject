# 11_UCF11 DataSet 파일 생성

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import random
import os

# ucf11 mpg 파일의 갯수
print(len(glob('./UCF11_updated_mpg/*/*/*.mpg')))

file_paths = glob('./UCF11_updated_mpg/*/*/*.mpg')

print(file_paths[0])

cap = cv2.VideoCapture(file_paths[0])

frames = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (256, 256))
    frame = frame[:, :, [2, 1, 0]]
    frames.appends(frame)
cap.release()

arr = np.array(frames)
plt.figure(figsize=(15, 15))
for i in range(10) :
    plt.subplot(10, 3, 1+3*i)
    plt.imshow(arr[1+3*i])
    plt.subplot(10, 3, 2+3*i)
    plt.imshow(arr[2+3*i])
    plt.subplot(10, 3, 3+3*i)
    plt.imshow(arr[3+3*i])
plt.tight_layout()

len(frames) / 29.97

df = pd.DataFrame(columns=['file_path', 'frames', 'duration', 'label'])

for file_path in file_paths :
    label = file_path.split('\\')[1]
    cap = cv2.VideoCapture(file_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / 29.97
    elem = {
        'file_path' : file_path,
        'frames' : frames,
        'duration' : duration,
        'label' : label
    }
    df.loc[len(df)] = elem
    cap.release()

df

# label 별 group 을 맺어 각 그룹의 영상의 길이를 출력
df_duration_sum_label = df.groupby('label').duration.sum().rename('sum')
df_duration_sum_label

# label 별 average(평균 영상 길이) 를 출력
df_duration_avg_label = df.groupby('label').duration.mean().rename('average')
df_duration_avg_label

df_video_states = pd.concat([df_duration_sum_label, df_duration_avg_label], axis=1)
df_video_states.plot.bar(secondary_y='Average')

label_dir = glob('UCF11_updated_mpg/*')
label_dir

# 11 개의 class 로 구성되어 있다.
# 각 class 당 25 개의 영상 그룹이 있다.
# 모델에 넣어 학습을 시킬 때, 0 ~ 19 까지는 학습그룹으로 구성한다.
# 20 ~ 24 까지는 test 그룹으로 구성한다.

train_df = pd.DataFrame(
    columns = ['file_path', 'label']
)

valid_df = pd.DataFrame(
    columns = ['file_path', 'label']
)

label_dirs = glob('UCF11_updated_mpg/*')

for label_dir in label_dirs :
    file_dirs = glob(label_dir + '\\v_*')
    random.shuffle(file_dirs)

    for i in range(20) :
        train_dir = file_dirs[i]
        label = train_dir.split('\\')[-1].split('_')[1] # 전체 경로에서 \\ 두개로 전부 끊고, 맨 끝의 이름을 _를 기준으로 split 하여 그 두 번째 이름을 가져온다.
        file_path = random.choice(glob(train_dir + '\\*')) # train_dir 안의 전체 경로 중 하나를 뽑아 사용
        train_df.loc[len(train_df)] = [file_path, label]

    for i in range(20, 25):
        valid_dir = file_dirs[i]
        label = valid_dir.split('\\')[-1].split('_')[
            1]  # 전체 경로에서 \\ 두개로 전부 끊고, 맨 끝의 이름을 _를 기준으로 split 하여 그 두 번째 이름을 가져온다.
        file_path = random.choice(glob(valid_dir + '\\*'))  # train_dir 안의 전체 경로 중 하나를 뽑아 사용
        valid_df.loc[len(valid_df)] = [file_path, label]

print(len(train_df)) # 220
print(len(valid_df)) # 55

train_df

os.mkdir('UCF11_updated_mpg')
os.mkdir('UCF11_updated_mpg/train')
os.mkdir('UCF11_updated_mpg/valid')

train_df.to_csv('ucf11_train_vid.csv', index=False)
valid_df.to_csv('ucf11_valid_vid.csv', index=False)

max_frame = 10
SAVE_DIR = 'UCF11_updated_png/'

for i, elem in train_df.iterrows() :
    cap = cv2.VideoCapture(
        elem['file_path']
    )

    frames = []
    while True :
        ret, frame = cap.read()

        if not ret :
            break

        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)

        if len(frames) == max_frame :
            break

    label = elem['label']
    for j, frame in enumerate(frames) :
        file_name = f'train/{label}_{i}_{j}.png'
        cv2.imwrite(SAVE_DIR + file_name, frame)

    cap.release()

print(len(glob(SAVE_DIR + 'train/*'))) # 2200

for i, elem in valid_df.iterrows() :
    cap = cv2.VideoCapture(
        elem['file_path']
    )

    frames = []
    while True :
        ret, frame = cap.read()

        if not ret :
            break

        frame = cv2.resize(frame, (256, 256))
        frames.append(frame)

        if len(frames) == max_frame :
            break

    label = elem['label']
    for j, frame in enumerate(frames) :
        file_name = f'valid/{label}_{i}_{j}.png'
        cv2.imwrite(SAVE_DIR + file_name, frame)

    cap.release()

print(len(glob(SAVE_DIR + 'train/*')))  # 550

# 12_Model training 파일 생성