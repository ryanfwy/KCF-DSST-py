import cv2
from tracker import KCFTracker

def tracker(cam, frame, bbox):
    tracker = KCFTracker(True, True, True) # (hog, fixed_Window, multi_scale)
    ok = tracker.init(bbox, frame)
    while True:
        ok, frame = cam.read()

        timer = cv2.getTickCount()
        bbox = tracker.update(frame)
        bbox = list(map(int, bbox))
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # Put FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    # ok, frame = video.read()
    ok, frame = video.read()
    bbox = cv2.selectROI('Select ROI', frame, False)

    if min(bbox) == 0: exit(0)
    tracker(video, frame, bbox)
