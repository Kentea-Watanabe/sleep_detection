# 参考:https://google.github.io/mediapipe/solutions/face_mesh.html
# 残タスク -> 居眠り回数が10回行ったら音声出力

"""
LandMark index memo 

目の上側の中心 : 159, 386
目の下側の中心 : 145, 374


"""
import math

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
delay = 1
frame = 0
sleep_count = 0

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # # 判定する基準の設定
    if results.multi_face_landmarks:
        # 右目
        right_eye_upper = results.multi_face_landmarks[0].landmark[159]
        right_eye_lower = results.multi_face_landmarks[0].landmark[145]
        # 2点間の距離を求める
        right_distance = math.sqrt((right_eye_upper.x - right_eye_lower.x)**2 + (right_eye_upper.y - right_eye_lower.y)**2)

        # 左目
        left_eye_upper = results.multi_face_landmarks[0].landmark[386]
        left_eye_lower = results.multi_face_landmarks[0].landmark[374]
        # 2点間の距離を求める
        left_distance = math.sqrt((left_eye_upper.x - left_eye_lower.x)**2 + (left_eye_upper.y - left_eye_lower.y)**2)

        # countを2秒目をつぶっていたら眠った判定を行う。
        if right_distance < 0.01 and left_distance < 0.01 :
            # print(frame)
            frame +=1
            if frame > 60:
                sleep_count += 1
                # frame数の初期化
                frame = 0
                print('sleep count', f'cnt : {sleep_count}')
        else:
            frame = 0

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.
    # cv2.putText(image, "居眠り回数:" + str(sleep_count), (0, 20),
    #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
    #                cv2.LINE_AA)
    cv2.putText(image,
            text='sleep count : ' + str(sleep_count),
            org=(10, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4)
    # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()