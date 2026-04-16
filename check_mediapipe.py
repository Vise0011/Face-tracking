import mediapipe as mp
import inspect

print('file:', mp.__file__)
print('version:', getattr(mp, '__version__', 'unknown'))
print('has solutions attribute:', hasattr(mp, 'solutions'))
print('solutions in dir:', 'solutions' in dir(mp))
print('sol* keys:', [k for k in dir(mp) if k.startswith('sol')])
print('module repr:', repr(mp))

try:
	import mediapipe.solutions as sols
	print('import mediapipe.solutions -> OK, repr:', repr(sols))
except Exception as e:
	print('import mediapipe.solutions -> FAILED:', type(e).__name__, e)

try:
	from mediapipe import solutions as sols2
	print('from mediapipe import solutions -> OK, repr:', repr(sols2))
except Exception as e:
	print('from mediapipe import solutions -> FAILED:', type(e).__name__, e)

print('\n--- modules and tasks (filesystem) ---')
import os
pkg_dir = os.path.dirname(mp.__file__)
print('package dir listing:', os.listdir(pkg_dir))
modules_dir = os.path.join(pkg_dir, 'modules')
tasks_dir = os.path.join(pkg_dir, 'tasks')
if os.path.isdir(modules_dir):
	print('modules dir listing:', os.listdir(modules_dir))
else:
	print('modules dir not present on filesystem')
if os.path.isdir(tasks_dir):
	print('tasks dir listing:', os.listdir(tasks_dir))
else:
	print('tasks dir not present on filesystem')
