from imutils.video import VideoStream
from imutils.video import FPS
import time
from mtcnn import MTCNN
import cv2

def show_results(img, face):
    h,w,c = img.shape
    # print(h,w,c)
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1, y1, face_width, face_height = face['box']
    x2, y2 = x1 + face_width, y1 + face_height

    # crop and output face
    crop_face = img[y1:y2,x1:x2]
    # print(y1,y2,x1,x2)
    # if crop_face.size != 0:
    #     output_file_name = output_path[:-4] + '_' + str(y1) + '_' + str(x1) + '.jpg'
    #     cv2.imwrite(output_file_name, crop_face)

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = {'nose': (255,0,0),
             'right_eye': (0,255,0),
             'left_eye': (0,0,255),
             'mouth_right': (255,255,0),
             'mouth_left': (0,255,255)}
    # nose, right-eye, left-eye, mouth-right, mouth-left

    for i in face['keypoints'].keys():
        cv2.circle(img, face['keypoints'][i], tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(face['confidence'])[:5]

    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def detect_webcam(detector):
    # Load model

    # initialize the video stream and allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()

    time.sleep(1.0)
    # start the FPS counter
    fps = FPS().start()
    i = 0
    num_faces = 0
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        frame = vs.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        for face in faces:
            frame = show_results(frame, face)
        # display the image to our screen
        cv2.imshow("Frame", frame)
        num_faces += len(faces)
        i += 1
        key = cv2.waitKey(1) & 0xFF
        # cv2.imwrite(output_path[:-4] +'_result.jpg', orgimg)
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or i == 100:
            break
        # update the FPS counter
        fps.update()
    
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print('num frames:', i)
    print('num faces', num_faces)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    detector = MTCNN()
    detect_webcam(detector)