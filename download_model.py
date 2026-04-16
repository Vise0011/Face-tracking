import urllib.request
import os

# Try multiple download sources
model_urls = [
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker.tflite',
    'https://github.com/google-ai-edge/mediapipe/releases/download/v0.10.5/face_landmarker.tflite',
]

out_path = os.path.join(os.path.dirname(__file__), 'models', 'face_landmarker_full.tflite')

print(f'Saving to {out_path}')
for model_url in model_urls:
    try:
        print(f'Trying {model_url}...')
        urllib.request.urlretrieve(model_url, out_path)
        sz = os.path.getsize(out_path)
        print(f'Success! Downloaded {sz} bytes')
        break
    except Exception as e:
        print(f'  Failed: {type(e).__name__}')
        if os.path.exists(out_path):
            os.remove(out_path)
        continue
else:
    print('All download sources failed.')
