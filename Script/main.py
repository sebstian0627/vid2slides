import cv2
import numpy as np
from timeit import timeit
import os
# import matplotlib.pyplot as plt


def make_histogram(l):
    d=dict()
    for i in l:
        d[i] = d.get(i,0)+1
    return d


def vid_imgs(file_path, folder_name='D:\\temp_project\\'):
    cap = cv2.VideoCapture(file_path)
    # os.mkdir(folder_name)
    # os.chdir(folder_name)
    if not cap.isOpened():
        print("ERROR, chutiya file hai")
    ret, frame = cap.read()
    frame_shape = frame.shape
    prev_frame = frame
    prev_difference = 0
    count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    temp_list=[]
    start = timeit()
    slide_count = 0
    while cap.isOpened():
        count += 1
        if count%14000==0:
            print(count/14)
        ret, frame = cap.read()
        if ret and count%fps == 0:
            temp1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp1 = temp1/(np.linalg.norm(temp1))
            temp2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            temp2 = temp2/(np.linalg.norm(temp2))
            print("HELLo")
            print("Another ")

            # print(type(temp1))
            alpha = np.sum(np.abs(temp1-temp2))
            temp_list.append(round(alpha))
            # if count >50:
            #     break
            if alpha > 20:
                # if count <50:

                    # cv2.imshow(str(slide_count),frame)
                cv2.imwrite(folder_name+ str(slide_count)+".jpg", frame)

                
                slide_count+=1
            prev_frame= frame
        elif not ret:
            break

        # print(count, "hello I am still here")
    end = timeit()
    cap.release()


    print(len(temp_list))
    # print(temp_list[0])
    print(make_histogram(temp_list))
    print("TOTAL TIME TAKEN", end-start)

    return make_histogram(temp_list)


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
    print("Hi I am here")
    out.release()
    cap.release()


if __name__ == '__main__':
    # print(os.getcwd())
    # vid_trimmer("D:\\downloads\\Link for class wednesday 12-01-22-20220112_110314-Meeting Recording.mp4")
    vid_imgs("D:\\downloads\\Link for class wednesday 12-01-22-20220112_110314-Meeting Recording.mp4")
    # print("hello I am completed")
    # os.chdir("D:\\downloads\\")
    # print((os.system('dir /OD')))