"""Synthetic fixtures only. Requires OpenCV 4.12, NumPy, Pillow and ONNX Runtime.

tensor() is the frozen v2.1 tensor recipe from frozen_kd.py, SHA256
986b9d58b8fd2a4426d3231b70bf59d3e40fa04c0c15415deb7e239af32785fb.
No training, calibration, threshold selection or medical images are used.
"""
from pathlib import Path
import gzip
import hashlib
import json
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/v21-fixtures"
MODEL = ROOT / "assets/offirst_v2_1_experimental.onnx"
MODEL_SHA = "b6bcb1182db7717c01ef24181d79a44be7d1a75ddefe8567cfec2fffb9e5c59b"
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def tensor(raw):
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid image")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]; scale = 512.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
    mask = (small.max(axis=2) > 15).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) >= 100:
        x0, x1 = int(xs.min()/scale), int((xs.max()+1)/scale)
        y0, y1 = int(ys.min()/scale), int((ys.max()+1)/scale)
        img = img[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]
        h, w = img.shape[:2]; m = max(h, w)
        pad = np.zeros((m,m,3),img.dtype)
        pad[(m-h)//2:(m-h)//2+h,(m-w)//2:(m-w)//2+w] = img
        img = pad
    cache = cv2.resize(img,(640,640),interpolation=cv2.INTER_AREA)
    work = cv2.resize(cache,(512,512),interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.ascontiguousarray(((work-MEAN*255)/(STD*255)).transpose(2,0,1))


def synthetic(size, color):
    im = Image.new('RGB', size)
    d = ImageDraw.Draw(im)
    w,h = size
    d.ellipse((w//8,h//10,w*7//8,h*9//10),fill=color)
    d.ellipse((w//3,h//4,w//2,h//2),fill=(230,170,100))
    d.line((w//2,h//2,w*3//4,h//3),fill=(70,25,15),width=max(2,w//100))
    return im


def main():
    assert cv2.__version__.startswith('4.12.'), cv2.__version__
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    for name,size,color,rotation in [
        ('gallery.jpg',(1024,768),(145,65,35),None),
        ('gallery.png',(777,513),(180,80,40),None),
        ('camera_like.jpg',(1920,1080),(110,55,30),None),
        ('rotated_exif6.jpg',(1024,768),(145,65,35),6),
        ('large_12mp.jpg',(4000,3000),(160,70,30),None),
    ]:
        im=synthetic(size,color)
        kwargs={}
        if name.endswith('.jpg'): kwargs.update(quality=95,subsampling=0)
        if rotation:
            exif=Image.Exif();exif[274]=rotation;kwargs['exif']=exif
        im.save(OUT/name,**kwargs)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(MODEL),options,providers=['CPUExecutionProvider'])
    rows=[]
    for path in sorted(OUT.glob('*')):
        if path.suffix not in ('.png','.jpg'): continue
        raw=path.read_bytes();x=tensor(raw)
        probability=float(session.run(['dr_probability'],{'fundus_image':x[None]})[0][0,0])
        tensor_bytes=x.astype('<f4').tobytes()
        (OUT/(path.name+'.f32.gz')).write_bytes(gzip.compress(tensor_bytes,mtime=0))
        rows.append(dict(name=path.name,image_sha256=hashlib.sha256(raw).hexdigest(),
            tensor_sha256=hashlib.sha256(tensor_bytes).hexdigest(),probability=probability,
            threshold=0.524373471736908,at_or_above=probability>=0.524373471736908))
    result=dict(synthetic=True,model_sha256=MODEL_SHA,opencv=cv2.__version__,onnxruntime=ort.__version__,
                input='fundus_image float32 [1,3,512,512]',output='dr_probability float32 [1,1]',fixtures=rows)
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
