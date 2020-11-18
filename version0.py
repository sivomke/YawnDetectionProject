import cv2
import dlib
import numpy as np

# load face detector

# option 1: opencv built-in Haar cascades
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
# option 2: dlib HOG+linear classifier

# option 3: DNN


# load face landmarks predictor

# option 1: using pre-trained dlib face landmarks predictor
# it will identify all 68 landmarks
landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# option 2: train custom dlib landmark predictor
# example: https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/
# it will be faster, it will detect only eyes and mouth

# for dlib face landmark predictor those are
# indices that correspond to upper and lower lips
# upper lip: 48, 49, 50, 51, 52, 53, 54, 64, 60,
#            61, 62, 63
# lower lip: 55, 56, 57, 58, 59,
#            67, 66, 65

dlib_upper_lip = [48, 49, 50, 51, 52, 53, 54,
                  60, 61, 62, 63, 64]
dlib_lower_lip = [55, 56, 57, 58, 59,
                  65, 66, 67]



# load frame
cap = cv2.VideoCapture("12-MaleGlasses.avi")

while True:
    ret, frame = cap.read()

    # if we fail to read the frame, we quit
    if not ret:
        print("The end")
        break

    # convert frame to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # equalize image histogram for image to normalize brightness
    # and increase contrast
    grey = cv2.equalizeHist(grey)

    # detect face
    faces = faceCascade.detectMultiScale(grey)

    """"""""""""""""""""""""""""""""""""""""""""""""
    """WHAT TO DO IF MULTIPLE FACES HAVE BEEN DETECTED
    WE WANT ONLY DRIVERS FACE"""
    """1. run for each of them"""
    """2. take the first "face" detected """
    """"""""""""""""""""""""""""""""""""""""""""""""

    # we draw a bounding rectangle for face area

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h),
                      color=(0, 255, 0),
                      thickness=2)


        # x - "columns"
        # y - "rows"
        # creating dlib rectangle containing face
        # top left corner (left, top)
        # bottom right corner (right, bottom)
        face_roi = dlib.rectangle(left=x,
                                  top=y,
                                  right=x+w,
                                  bottom=y+h)

        # getting 68 face landmarks
        landmarks = landmarks_predictor(image=grey,
                                        box=face_roi)
        """
        # drawing landmarks
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            cv2.circle(img=frame,
                       center=(x, y),
                       radius=1,
                       color=(0, 255, 0),
                       thickness=2)
        """



        # selecting and drawing landmarks that correspond
        # to upper and lower lip
        for i in dlib_upper_lip:
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            cv2.circle(img=frame,
                       center=(x, y),
                       radius=1,
                       color=(0, 255, 0),
                       thickness=2)

        for j in dlib_lower_lip:
            x = landmarks.part(j).x
            y = landmarks.part(j).y

            cv2.circle(img=frame,
                       center=(x, y),
                       radius=1,
                       color=(0, 255, 0),
                       thickness=2)

    # calculate the distance between centers of upper and lower lip
    # 51 - center of upper lip
    # 57 - center of lower lip
    dist = np.sqrt((landmarks.part(51).x - landmarks.part(57).x)**2 +
                   (landmarks.part(51).y - landmarks.part(57).y)**2)
    print(f"distance: {np.round(dist, 2)}")

    # update score
    # we calculate number of consecutive frames where the distance
    # exceeds the threshold (score) (IT SHOULD DEPEND ON FPS),
    # if it is bigger than some number -> alert
    # if the distance is less than the threshold we set
    # the score to zero

    # show the frame
    cv2.imshow("Frame", frame)

    # we can stop by pressing "q" key
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()










