import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time
import keyboard 

MODEL_PATH = 'runs/detect/cookie_detector-4/weights/bestV1.onnx' 
CONFIDENCE_THRESHOLD = 0.6
CLICKS_PER_OBJECT = 10 
CLICK_DELAY = 0.1 

print("Бот запущен! Нажмите ПРОБЕЛ для экстренной остановки.")

model = YOLO(MODEL_PATH)

def find_and_click():
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    results = model(screenshot_np, verbose=False)
    
    clicked_something = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]
            
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                for i in range(CLICKS_PER_OBJECT):
                    pyautogui.click(center_x, center_y)
                    time.sleep(CLICK_DELAY)
                
                print(f"✅ Кликнул по: {class_name} (Уверенность: {confidence:.2f}) | Координаты: ({center_x}, {center_y})")
                clicked_something = True
                
    return clicked_something

while True:
    if keyboard.is_pressed('space'):
        print("🛑 Бот остановлен пользователем (нажат пробел).")
        break
    
    if find_and_click():
        time.sleep(1) 
    else:
        time.sleep(0.5) 
