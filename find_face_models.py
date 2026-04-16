import mediapipe as mp
import os
pkg_dir = os.path.dirname(mp.__file__)
res = []
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        if 'face' in f.lower() and f.lower().endswith(('.tflite', '.pb')):
            res.append(os.path.join(root, f))
print('package dir:', pkg_dir)
print('found models:', res)
