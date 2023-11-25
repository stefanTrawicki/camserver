import cv2, time
from threading import Thread
from queue import Queue
from dataclasses import dataclass
import numpy as np
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
target_fps = 10
frames_between_capture = 30
img_counter = 0
frames = 0
frame_queue = Queue()

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

@dataclass
class QueuedFrame:
    n: int
    data: np.ndarray
    timestamp: float

# def idk(image):
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#         )

def frame_processor():
    while True:
        frame = frame_queue.get()

        print(f"Processing frame {frame.n} ({frame.timestamp})")
        file = f"opencv_frame_{frame.n}".format(img_counter)

        image = Image.fromarray(frame.data, "RGB")
        
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        box = [round(i, 2) for i in results["boxes"][0].tolist()]
        score = results["scores"][0]
        label = results["labels"][0]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(box), outline=(255, 255, 255))
        image = Image.fromarray(np.array(image)[:,:,::-1])
        image.save(f"{file}.png", "PNG")
        print(f"Wrote {file}.png to disk!")


Thread(target=frame_processor, daemon=True).start()

while True:
    start = time.time()
    ret, frame = cam.read()
    cv2.imshow("test", frame)

    # push frame for processing frame

    frames += 1
    if frames % frames_between_capture == 0:
        frame_queue.put(QueuedFrame(img_counter, frame, time.time()))
        img_counter += 1
        frames = 0

    k = cv2.waitKey(1) % 256
    if k == 27:
        print("Escape hit, closing...")
        break

    end = time.time() - start
    if (1/target_fps) - end > 0:
        time.sleep((1/target_fps) - end)

cam.release()

cv2.destroyAllWindows()