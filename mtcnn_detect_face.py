from imutils.video import VideoStream
from imutils.video import FPS
import time
from mtcnn import MTCNN
import cv2

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
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        num_faces += len(faces)
        i += 1
        key = cv2.waitKey(1) & 0xFF
        # cv2.imwrite(output_path[:-4] +'_result.jpg', orgimg)
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or i == 10:
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