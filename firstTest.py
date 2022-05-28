import cv2 as cv
import mediapipe as mp
import time
import utils, math

#test
testString = "0 "
f = open("NoLookTesting", 'w')

# variables 
frame_counter =0
CEF_COUNTER =0
JAWNING_COUNTER=0
TOTAL_BLINKS =0
fps=0

# constants
CLOSED_EYES_FRAME = 1
FONTS =cv.FONT_HERSHEY_COMPLEX


LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS=[474, 475, 476, 477]
RIGHT_IRIS=[469, 470, 471, 472]
LIPS=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

map_face_mesh = mp.solutions.face_mesh

camera = cv.VideoCapture("DriverMonitoring\mypaperdataset.mp4")
time.sleep(1)

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def IsBlinking(landmarks, right_indices, left_indices):
    # Right eye
    # vertical line 
    rightTop = landmarks[right_indices[12]]
    rightBottom = landmarks[right_indices[4]]

    # LEFT_EYE 
    # vertical line 
    leftTop = landmarks[left_indices[12]]
    leftBottom = landmarks[left_indices[4]]

    #horizontal
    horizontalRight=landmarks[right_indices[0]]
    horizontalLeft=landmarks[left_indices[8]]
    horizontalDistance=euclaideanDistance(horizontalLeft,horizontalRight)

    #vertical total
    rightEye = euclaideanDistance(rightTop, rightBottom)
    leftEye = euclaideanDistance(leftTop, leftBottom)
    verticalDistance=rightEye+leftEye

    if verticalDistance/horizontalDistance<0.1:
        return True
    return False


def IsJawning(landmarks, mouth_indicies):

    #horizontal
    mouth_right = landmarks[mouth_indicies[0]]
    mouth_left = landmarks[mouth_indicies[10]]

    # vertical
    mouth_top = landmarks[mouth_indicies[15]]
    mouth_bottom = landmarks[mouth_indicies[5]]
    
    horizontalDistance = euclaideanDistance(mouth_left,mouth_right)
    verticalDistance = euclaideanDistance(mouth_top,mouth_bottom)

    ratio=horizontalDistance/verticalDistance
    if ratio<1.5:
        return True
    return False

def lookingDirection(landmarks, left_iris, right_iris, left_eye, right_eye):
    #RIGHT
    rightIris_right = landmarks[right_iris[2]]
    rightIris_left = landmarks[right_iris[0]]

    rightEye_right = landmarks[right_eye[0]]
    rightEye_left = landmarks[right_eye[8]]

    # LEFT
    leftIris_right = landmarks[left_iris[2]]
    leftIris_left = landmarks[left_iris[0]]

    leftEye_right = landmarks[left_eye[0]]
    leftEye_left = landmarks[left_eye[8]]

    right_r=euclaideanDistance(rightEye_right,rightIris_right)
    left_r=euclaideanDistance(leftEye_right, leftIris_right)

    right_l=euclaideanDistance(rightEye_left,rightIris_left)
    left_l=euclaideanDistance(leftEye_left, leftIris_left)
    
    rightDistance=right_r+left_r
    leftDistance=right_l+left_l


    if abs(rightDistance-leftDistance)<20:
        direction="Center"
    elif rightDistance>leftDistance:
        direction="Right"
    else:
        direction="Left"

    return direction

with map_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence =0.5, min_tracking_confidence=0.8) as face_mesh:

    start_time = time.time()
    while True:
        
        frame_counter +=1
        ret, frame = camera.read()
        if not ret: 
            break
        
        frame = cv.flip(frame,1)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)

            if IsBlinking(mesh_coords,RIGHT_EYE,LEFT_EYE):
                CEF_COUNTER +=1

                #test
                #testString += " 1"

                if CEF_COUNTER>fps*0.5:
                    utils.colorBackgroundText(frame,  f'Warning!', FONTS, 2, (30, 450), 2, utils.RED, pad_x=6, pad_y=6, )
                if CEF_COUNTER>fps*2:
                    utils.colorBackgroundText(frame,  f'ALARM!', FONTS, 2, (400, 450), 2, utils.RED, pad_x=10, pad_y=10, )
                else:
                  utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (30, 200), 2, utils.YELLOW, pad_x=6, pad_y=6, )
            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0
            
            if IsJawning(mesh_coords,LIPS):
                        JAWNING_COUNTER += 1

                        if JAWNING_COUNTER>fps*0.6:

                            #test
                            #testString+=" 2"

                            utils.colorBackgroundText(frame,  f'Jawn', FONTS, 0.9, (30,260),2, utils.GREEN, utils.GRAY)
        else:
            utils.colorBackgroundText(frame,  f'NO DRIVER!', FONTS, 3, (20, 300), 2, utils.PURPLE, pad_x=10, pad_y=10, )
            testString += " 3"

        end_time = (time.time()-start_time)
        fps = frame_counter/end_time

        direction=lookingDirection(mesh_coords,LEFT_IRIS,RIGHT_IRIS,LEFT_EYE,RIGHT_EYE)
        utils.colorBackgroundText(frame,  f'Looking direction: {direction}', FONTS, 0.7, (30,80),2, utils.BLUE, utils.GRAY)
       
        #fps regulation
        while fps>23.976:
            time.sleep(0.0000000001)
            fps = frame_counter/(time.time()-start_time)

        #test
        if direction!="Center":
            testString += " 3"

        f.write( "{}".format(float(f'{end_time:.3f}')) +" "+ testString +"\n")
        testString = "{} ".format(frame_counter)
        utils.colorBackgroundText(frame,  f'FPS: {fps}', FONTS, 0.7, (1250,90),2, utils.YELLOW, utils.GRAY)


        blinkFrequency = TOTAL_BLINKS/end_time
        utils.colorBackgroundText(frame,  f'Blink frequency: {round(blinkFrequency,2)}', FONTS, 0.7, (30,300),2, utils.ORANGE, utils.GRAY)
        utils.colorBackgroundText(frame,  f'Total blnks: {round(TOTAL_BLINKS,2)}', FONTS, 0.7, (30,330),2, utils.PINK, utils.GRAY)

        cv.imshow('Driver Monitoring', frame)
        key = cv.waitKey(1)
        if key==ord('q') or key ==ord('Q'):
            break
    

    cv.destroyAllWindows()
    camera.release()