import os
import cv2
import numpy as np

try:
    import mediapipe as mp
    # Prefer legacy `mp.solutions` API when available
    if hasattr(mp, 'solutions'):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        cap = cv2.VideoCapture(0)

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            print("웹캠을 시작합니다. 종료하려면 'ESC' 키를 누르세요.")

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("웹캠을 찾을 수 없습니다.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = face_mesh.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                cv2.imshow('AI-Vtuber Vision Test', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()
    else:
        raise AttributeError('no mp.solutions')
except Exception as e:
    # Fallback for MediaPipe 0.10+ (tasks API)
    print(f'mp.solutions unavailable ({type(e).__name__}), using Tasks API...')
    
    import mediapipe as mp
    from mediapipe.tasks.python.vision import face_landmarker as fl
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe.tasks.python.vision.core import image as mp_image

    # Path to a Face Landmarker TFLite model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'face_landmarker_full.tflite')

    if not os.path.exists(MODEL_PATH):
        print('ERROR: Cannot find Face Landmarker model at', MODEL_PATH)
        print('To use Tasks API, download the model from:')
        print('  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker.tflite')
        print('Place it at:', MODEL_PATH)
        raise SystemExit(1)

    # [핵심 수정] 한글 경로 문제를 피하기 위해 파일을 직접 읽어서 데이터로 넘깁니다.
    print(f"모델 로딩 중: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model_data = f.read()

    # Create options with model buffer (파일 경로 대신 데이터를 사용)
    base_options = BaseOptions(model_asset_buffer=model_data)
    options = FaceLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE)
    
    # Create landmarker
    landmarker = fl.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("웹캠을 시작합니다. 종료하려면 'ESC' 키를 누르세요.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print('웹캠을 찾을 수 없습니다.')
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)

        result = landmarker.detect(mp_img)

        h, w, _ = frame.shape
        if result and getattr(result, 'face_landmarks', None):
            for face in result.face_landmarks:
                pts = []
                for lm in face:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # draw connections
                conns = fl.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
                for c in conns:
                    s, e = c.start, c.end
                    if s < len(pts) and e < len(pts):
                        cv2.line(frame, pts[s], pts[e], (0, 128, 255), 1)

        cv2.imshow('AI-Vtuber Vision Test (tasks API)', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()