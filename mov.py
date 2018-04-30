import cv2

fourcc = cv2.VideoWriter_fourcc(* 'MP4S')
video = cv2.VideoWriter('test.mp4', fourcc, 1, (320, 240))

for i in range(1, 21):
    img = cv2.imread('a (' + str(i) + ').jpg')
    img = cv2.resize(img, (320,240))
    video.write(img)

video.release()