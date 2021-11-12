import pyvirtualcam
import numpy as np

import mediapipe as mp

from videoclass import *  # threads

# cap= cv2.VideoCapture(0)
DimX= 1280
DimY= 720
cap = VideoStream(1, x=DimX, y=DimY).start()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils
drawing_spec = mpDraw.DrawingSpec(color=[255,0,0],thickness=2, circle_radius=2)


with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
    while True:
        img = cap.read()
        black_image = np.zeros((DimY, DimX, 3), np.uint8)
        #h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)

        if results.multi_face_landmarks:
            for  faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(imgRGB, faceLms, mp_face_mesh.FACE_CONNECTIONS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)
            cx_min = DimX
            cy_min = DimY
            cx_max = cy_max = 0
            for _, lm in enumerate(faceLms.landmark):
                cx, cy = int(lm.x * DimX), int(lm.y * DimY)
                if cx < cx_min:
                    cx_min = cx
                if cy < cy_min:
                    cy_min = cy
                if cx > cx_max:
                    cx_max = cx
                if cy > cy_max:
                    cy_max = cy
            cv2.rectangle(imgRGB, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 1)
            cv2.putText(imgRGB, "G. Suanno", (cx_min, cy_min-8), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 255), 2)

        #cv2.imshow('img', imgRGB)
        cam.send(imgRGB)
        cam.sleep_until_next_frame()
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break  # wait for ESC key to exit


    # cap.release()
    cap.stop()
    cv2.destroyAllWindows()

