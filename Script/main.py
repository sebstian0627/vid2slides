import cv2
import numpy as np
from time import time
import os
from PIL import Image
# import matplotlib.pyplot as plt


def make_histogram(l):
    d=dict()
    for i in l:
        d[i] = d.get(i,0)+1
    return d


def vid_imgs(file_path, folder_name='D:\\temp_project\\', thresh=20):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("ERROR! Corrupted Video")
    ret, frame = cap.read()
    frame_shape = frame.shape
    prev_frame = frame
    prev_difference = 0
    count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    temp_list=[]
    start = time()
    slide_count = 0
    img_list=[]
    while cap.isOpened():
        count += 1
        # print(count)
        if count%14000==0:
            print(count/14)
        ret, frame = cap.read()
        if ret and count%fps == 0:
            temp1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp1 = temp1/(np.linalg.norm(temp1))
            temp2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            temp2 = temp2/(np.linalg.norm(temp2))
            alpha = np.sum(np.abs(temp1-temp2))
            temp_list.append(round(alpha))
            
            if alpha > thresh:
                

                img_list.append(frame)

                
                slide_count+=1
            prev_frame= frame
        elif not ret:
            break

        # print(count, "hello I am still here")
    end = time()
    cap.release()


    print(len(temp_list))
    print(make_histogram(temp_list))
    print("TOTAL TIME TAKEN", end-start)

    return img_list


def vid_trimmer(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("ERROR, chutiya file hai")
    ret, frame = cap.read()
    h, w, _ = frame.shape
    frame_shape = (h,w)
    count = 0
    print(int(cap.get(cv2.CAP_PROP_FPS)))
    out = cv2.VideoWriter('D:\\downloads\\trimmed.avi', cv2.VideoWriter_fourcc(*'XVID'), 14, (1080, 1920))
    while cap.isOpened():
        count += 1
        print(count)
        ret, frame = cap.read()
        if ret:
            # frame = cv2.flip(frame, 0)
            print(frame.shape)
            out.write(frame)
        if count == 10000:
            break
    out.release()
    cap.release()

def images2pdf(img_list,pdf_path,pdf_name):
    # if(img_list==None or len(img_list)==0):
    #     print("wrong data type, please check input carefully")
    #     return
    # print(img_list)
    if len(img_list)==1:
        img_list[0] = img_list[0].convert('RGB')
        img_list[0].save(pdf_path + pdf_name)
    img_list1 = [Image.fromarray(i) for i in img_list]
    # img_list1 = [i.convert('RGB') for i in img_list1]
    im1 = img_list1[0]
    im1.save(pdf_path+pdf_name,save_all=True, append_images = img_list1[1:])
    print("ALL IMAGES ARE SAVED SUCCESSFULLY AS PDF AT", pdf_path)



if __name__ == '__main__':
    # print(os.getcwd())
    # vid_trimmer("D:\\downloads\\Link for class wednesday 12-01-22-20220112_110314-Meeting Recording.mp4")
    temp = vid_imgs("D:\\downloads\\Link for class wednesday 12-01-22-20220112_110314-Meeting Recording.mp4")
    images2pdf(temp, "D:\\", "first.pdf")
    print("hello")
    