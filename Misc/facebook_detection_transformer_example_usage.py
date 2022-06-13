import warnings
import requests
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


warnings.filterwarnings('ignore')

URL = 'https://www.zr.ru/d/story/d9/921305/tass_36084299.jpg'
TOP_K_CLASSES = 20
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

model = torch.hub.load(
    'facebookresearch/detr', 'detr_resnet50', pretrained=True
)
model.eval()
model.cuda()

image_original = Image.open(requests.get(URL, stream=True).raw)

tt = transforms.Compose([
    transforms.ToTensor(),
])
image = tt(image_original).unsqueeze(0).cuda()

with torch.no_grad():
    output = model(image)

predicted_labels = output['pred_logits'][0]
predicted_bboxes = output['pred_boxes'][0]
max_predicted_labels = predicted_labels[:, :len(CLASSES)].softmax(-1).max(-1)
topk_to_show = max_predicted_labels.values.topk(TOP_K_CLASSES)
predicted_labels = predicted_labels[topk_to_show.indices]
predicted_bboxes = predicted_bboxes[topk_to_show.indices]

img = image_original.copy()
draw = ImageDraw.Draw(img)
orig_height, orig_width = image.shape[-1], image.shape[-2]

for label, bbox in zip(predicted_labels, predicted_bboxes):
    label = label.argmax()
    if label >= len(CLASSES):
        continue
    class_detected = CLASSES[label]
    bbox = bbox.cpu() * torch.Tensor(
        [orig_height, orig_width, orig_height, orig_width]
    )
    x, y, w, h = bbox
    x0, x1 = x - w // 2, x + w // 2
    y0, y1 = y - h // 2, y + h // 2
    draw.rectangle([x0, y0, x1, y1], outline='red')
    draw.text((x, y), class_detected, fill='red')
img.show()
img.save('processed_image.jpg')
