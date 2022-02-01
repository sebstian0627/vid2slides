import cv2
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

def make_histogram(l):
    d=dict()
    for i in l:
        d[i] = d.get(i,0)+1
    return d
