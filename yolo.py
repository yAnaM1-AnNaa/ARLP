import json
from ultralytics import YOLO
import pandas as pd

# Load a model
model = YOLO("yolo11x.pt")  # pretrained YOLO11n model
# 在终端输出 DataFrame 并保存为excel文件
# df = pd.DataFrame(list(model.names.items()), columns=['Class ID', 'Class Name'])
# pd.set_option('display.max_rows', None)
# print(df)
# df.to_excel("yolo_classes.xlsx", index=False)
img = r"E:\GithubRepository\ProjectX\dataset\test\bed.jpg"
obj = img.split('\\')[-1].split('.')[0]
json_path = f"E:\\GithubRepository\\ProjectX\\dataset\\testresult\\{obj}.json"
results = model([img], classes=[59])  # return a list of Results objects，也可以用[img1, img2, img3]的形式传入多张图片

# Process results list
predictions = []
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    for box in boxes:
        coords = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])
        predictions.append({
            "bbox": coords, # [x1, y1, x2, y2]
            "label": class_name,
            "confidence": conf
        })
    save_path = rf"dataset\testresult\result_{obj}.jpg"
    result.save(filename=save_path) 
    print(f'processed image saved to {save_path}.')

with open(json_path, 'w') as f:
    json.dump(predictions, f, indent=4)
print(f"Saved {len(predictions)} detected objects to {json_path}")
