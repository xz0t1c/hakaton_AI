import logging
import os
import cv2
import time
import asyncio
from imageai.Detection import ObjectDetection
from voice import object_voice
import datetime

some_bytes = b'\x01\x02'
logger = logging.getLogger()
exe_path = os.getcwd()
video_file_path = "00_43_47.mp4"  # Укажите путь к вашему видеофайлу
check_array = ['traffic light', 'person', 'car']
x1, y1, x2, y2 = 0, 0, 0, 0


async def process_frame(detector, frame):
    _, array_detection = await asyncio.to_thread(
        detector.detectObjectsFromImage,
        input_image=frame,
        output_type='array',
        minimum_percentage_probability=30
    )

    for obj in array_detection:
        if obj['name'] in check_array:
            # Извлекаем координаты прямоугольника
            global x1, y1, x2, y2
            x1, y1, x2, y2 = obj['box_points']
            filename = datetime.datetime.now().strftime("%d.%m.%Y %H-%M-%S") + ".png"
            cv2.imwrite(f'output_image_{filename}', frame)
            await object_voice(obj['name'])


async def main():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(exe_path, 'yolov3.pt'))
    detector.loadModel()
    detector.useCPU()
    finish = 0

    # Открываем видеофайл для чтения
    video_capture = cv2.VideoCapture(video_file_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        start = time.time()
        if start - finish > 1:
            asyncio.create_task(process_frame(detector, frame))
            finish = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('hakaton', frame)
        await asyncio.sleep(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
