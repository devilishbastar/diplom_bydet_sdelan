import asyncio
import base64
import json
import queue
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from utils import SLInference

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный инференс-поток (один экземпляр, общий для всех клиентов)
inference_thread = SLInference("configs/config.json")
inference_thread.start()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    gestures_deque = deque(maxlen=5)

    try:
        while True:
            # Принимаем бинарные данные кадра (JPEG)
            data = await websocket.receive_bytes()

            # Декодируем в numpy array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
            if frame is None:
                continue

            # Приводим к RGB и изменяем размер до 224x224 (соответствует вашей модели)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))

            # Помещаем в вашу очередь SLInference
            inference_thread.input_queue.append(frame_resized)

            # Получаем текущее предсказание
            gesture = inference_thread.pred

            # Логика дедупликации (аналогична вашему app.py)
            if gesture not in ['no', '']:
                if not gestures_deque:
                    gestures_deque.append(gesture)
                elif gesture != gestures_deque[-1]:
                    gestures_deque.append(gesture)

            # Отправляем результат клиенту
            await websocket.send_text(json.dumps({
                "gesture": gesture,
                "history": list(gestures_deque)
            }))

    except WebSocketDisconnect:
        print("Клиент отключился")
    except Exception as e:
        print(f"Ошибка: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)