import cv2
import mediapipe
import pyautogui

face_mesh_landmark = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam = cv2.VideoCapture(0)
screen_w,screen_h=pyautogui.size()
while True:
    _,image = cam.read()
    image = cv2.flip(image,1)
    #window_h,window_w,window_d=image.shape
    window_h,window_w,_=image.shape   #if we dont want depth use underscore

    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmark.process(rgb_image)
    all_face_landmarks = processed_image.multi_face_landmarks
    if all_face_landmarks:
        one_face_landmark = all_face_landmarks[0].landmark
        for id,landmark_point in enumerate(one_face_landmark[474:478]):
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            #print(x,y)
            if id==1:
                mouse_x = int(screen_w / window_w * x)
                mouse_y = int(screen_h / window_h * y)
                pyautogui.moveTo(mouse_x,mouse_y)
            cv2.circle(image,(x,y),3,(0,0,255))
        left_eye_landmark = (one_face_landmark[145],one_face_landmark[159])
        for landmark_point in left_eye_landmark:
            x = int(landmark_point.x * window_w)
            y = int(landmark_point.y * window_h)
            #print(x,y)
            cv2.circle(image,(x,y),3,(0,255,255))
        if(left_eye_landmark[0].y - left_eye_landmark[1].y<0.01):
            pyautogui.click()
            pyautogui.sleep(2)
            print('mouse clicked')

    cv2.imshow("Eye Controll Mouse",image)
    key = cv2.waitKey(100)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()