import cv2
import mediapipe as mp
from flask import Flask, jsonify
import threading

app = Flask(__name__)


num_landmarks = 0
last_data = None

@app.route('/')
def get_params():
  return jsonify(last_data)


def mediapipe_thread():
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_face_mesh = mp.solutions.face_mesh


  # For webcam input:
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  cap = cv2.VideoCapture(0)
  with mp_face_mesh.FaceMesh(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        global num_landmarks
        num_landmarks = len(results.multi_face_landmarks)
        for face_landmarks in results.multi_face_landmarks:

          last_data_local = []
          for landmark in face_landmarks.landmark:
            last_data_local.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
          global last_data
          last_data = last_data_local

          # print(face_landmarks)
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
      cv2.imshow('MediaPipe FaceMesh', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()



t = threading.Thread(target=mediapipe_thread)
t.start()

if __name__ == "__main__":
  app.run(port=8008)