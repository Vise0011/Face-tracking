from mediapipe.tasks.python.vision import face_landmarker
import inspect

print('module file:', face_landmarker.__file__)
print('members sample:', [k for k in dir(face_landmarker) if not k.startswith('_')][:200])
print('\n--- source start (truncated) ---\n')
print(inspect.getsource(face_landmarker)[:3000])
print('\n--- source end (truncated) ---\n')
