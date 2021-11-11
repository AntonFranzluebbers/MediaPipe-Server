import cv2
import mediapipe as mp
import threading
import asyncio
import socket
import json

num_landmarks = 0
last_data = None
# web_stream_url = 'https://vdo.ninja/?view=anton_op5t'
web_stream_url = 'https://vdo.ninja/?view=anton_win10'

webcam_index = 1
use_webcam = True
use_web_stream = False
use_static_image = False



# https://stackoverflow.com/questions/46932654/udp-server-with-asyncio

HOST, PORT = '192.168.5.148', 8642
# HOST, PORT = 'localhost', 8642




# Message to send to UDP port
def send_test_message(message) -> None:
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.sendto(message.encode(), (HOST, PORT))

# Continuously write messages to UDP port
async def write_messages():
    print("writing")
    while True:
        await asyncio.sleep(.1)
        if last_data is not None:
          send_test_message(json.dumps(last_data))


def mediapipe_thread():
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_holistic = mp.solutions.holistic


  # For webcam input:
  if use_webcam:
    cap = cv2.VideoCapture(webcam_index)
  elif use_web_stream:
    cap = cv2.VideoCapture(web_stream_url)
  elif use_static_image:
    # This isn't actually used, but used for while loop to work
    cap = cv2.VideoCapture(0)
  with mp_holistic.Holistic(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
      if use_webcam or use_web_stream:
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
      else:
        image = cv2.imread('person_jumping.jpg')

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image)

      global last_data
      last_data = {
        'face': [      {'x': -landmark.x, 'y': -landmark.y, 'z': -landmark.z} for landmark in (results.face_landmarks.landmark if results.face_landmarks else [])],
        'left_hand': [ {'x': -landmark.x, 'y': -landmark.y, 'z': -landmark.z} for landmark in (results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])],
        'right_hand': [{'x': -landmark.x, 'y': -landmark.y, 'z': -landmark.z} for landmark in (results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])],
        'body': [      {'x': -landmark.x, 'y': -landmark.y, 'z': -landmark.z} for landmark in (results.pose_world_landmarks.landmark if results.pose_world_landmarks else [])],
        # 'body_occl': [ {'x': -landmark.x, 'y': -landmark.y, 'z': -landmark.z} for landmark in (results.pose_world_landmarks.landmark if results.pose_world_landmarks else [])],
      }

      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image,
          results.face_landmarks,
          mp_holistic.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles
          .get_default_pose_landmarks_style())
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break
    print('fail')
  cap.release()



t = threading.Thread(target=mediapipe_thread)
t.start()

if __name__ == "__main__":
  loop = asyncio.get_event_loop()
  loop.run_until_complete(write_messages()) # Start writing messages (or running tests)
  loop.run_forever()