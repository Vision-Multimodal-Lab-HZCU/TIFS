import os
from pathlib import Path
import numpy as np
import cv2

def getfiles(dir):
    filenames=os.listdir(dir)
    return filenames

def merge(dir):
    video1_dir = 'dataset/' + str(dir) + '/' + 'video.mp4'
    video2_dir = 'dataset/' + str(dir) + '/' + 'gelsight.mp4'

    cap1 = cv2.VideoCapture(video1_dir)
    frame_number1 = int(cap1.get(7))

    cap2 = cv2.VideoCapture(video2_dir)
    frame_number2 = int(cap2.get(7))

    frame_number1 = min(frame_number1, frame_number2)
    # if processed previously, skip this video
    if os.path.exists('video_frame' +'/'+ str(dir) ):
        return
    if not os.path.exists('video_frame' +'/'+ str(dir)):
        os.makedirs('video_frame' +'/'+ str(dir))
    if not os.path.exists('gelsight_frame' +'/'+ str(dir)):
        os.makedirs('gelsight_frame' +'/'+ str(dir))

    for i in range(frame_number1):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        cv2.imwrite('video_frame' +'/'+ str(dir) +'/'+ str(i).rjust(10,'0') + '.jpg', frame1)
        cv2.imwrite('gelsight_frame' +'/'+ str(dir) +'/'+ str(i).rjust(10,'0') + '.jpg', frame2)

    cap1.release()
    cap2.release()

def main():
    dir = Path('dataset/')
    files = getfiles(dir)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for i in range(len(files)):
        fdir = files[i]
        print(str(fdir) + " Start!")
        merge(fdir)
        print(str(fdir) + " Finished!")

if __name__ == '__main__':
    main()
