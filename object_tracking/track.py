import torch
import cv2
import numpy as np
from Siam import SiameNN, cross_relation
import torchvision.transforms as T
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def track_video(video_path, ann_path, model_path, output_path):

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = SiameNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create output video {output_path}")
        cap.release()
        return

    # 导入第一帧选定的框
    ann = np.loadtxt(ann_path)
    if ann.ndim == 1:
        ann = ann[np.newaxis, :]
    x, y, w, h = ann[0] 
    cx, cy = x + w/2, y + h/2 

    # 对每一帧进行追踪，最后直接生成视频
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        out.release()
        return
    z = frame[int(max(0, y)):int(y+h), int(max(0, x)):int(x+w)]
    z = cv2.resize(z, (127, 127))
    z = transform(z).unsqueeze(0).to(device) # 加上一个batch-size这一个维度，这里默认一个batch就是1

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        size = int(max(w, h) * 2.0)
        x1, y1 = int(cx - size/2), int(cy - size/2)
        x2, y2 = x1 + size, y1 + size
        search = frame[max(0, y1):y2, max(0, x1):x2]

        search = cv2.resize(search, (255, 255))
        search = transform(search).unsqueeze(0).to(device)

        with torch.no_grad(): # 因为是推理模型，所以不用计算梯度，只用前向传播
            fz = model(z)
            fx = model(search)
            response = cross_relation(fz, fx).squeeze().cpu().numpy()

        max_idx = np.unravel_index(np.argmax(response), response.shape)
        scale = size / 255 
        new_cx = x1 + (max_idx[1] * scale)
        new_cy = y1 + (max_idx[0] * scale)
        new_x = int(new_cx - w/2)
        new_y = int(new_cy - h/2)

        cx, cy = new_cx, new_cy

        cv2.rectangle(frame, (new_x, new_y), (new_x + int(w), new_y + int(h)), (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Tracking completed. Output saved to {output_path}")

if __name__ == '__main__':
    video_path = 'data/videos/test/sample.mp4'
    ann_path = 'data/annotations/test/sample.txt'
    model_path = 'sia_best.pth'
    output_path = 'output/sample_tracked.mp4'
    track_video(video_path, ann_path, model_path, output_path)