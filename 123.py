
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from constants import classes

MODEL_PATH = "SignFlow-R.onnx"

FRAME_SIZE = 224
WINDOW_SIZE = 32
FRAME_INTERVAL = 2

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def preprocess_frame(frame):
    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Сохраняем пропорции и добавляем поля
    height, width = frame.shape[:2]
    scale = min(FRAME_SIZE / width, FRAME_SIZE / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    frame = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    pad_width = FRAME_SIZE - new_width
    pad_height = FRAME_SIZE - new_height

    left = int(round(pad_width / 2 - 0.1))
    right = pad_width - left
    top = int(round(pad_height / 2 - 0.1))
    bottom = pad_height - top

    frame = cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    frame = frame.astype(np.float32)
    frame = (frame - MEAN) / STD

    # HWC -> CHW
    frame = np.transpose(frame, (2, 0, 1))

    return frame


def main():
    print("Загрузка модели...")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print("Вход модели:", input_info.name, input_info.shape)
    print("Выход модели:", output_info.name, output_info.shape)
    print("Количество классов:", len(classes))

    if len(classes) != output_info.shape[-1]:
        raise ValueError(
            "Количество классов не совпадает с выходом модели!"
        )

    input_name = input_info.name
    output_name = output_info.name

    # Храним только кадры, которые будут переданы модели
    frames = deque(maxlen=WINDOW_SIZE)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть веб-камеру.")
        return

    frame_counter = 0
    current_word = "Покажите жест"
    confidence = 0.0
    last_prediction_time = 0.0

    print("Камера запущена. Для выхода нажмите Q.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Не удалось получить кадр с камеры.")
                break

            frame_counter += 1

            # Используем каждый второй кадр
            if frame_counter % FRAME_INTERVAL == 0:
                processed = preprocess_frame(frame)
                frames.append(processed)

            # Предсказываем после накопления 32 кадров
            if len(frames) == WINDOW_SIZE:
                # (C, T, H, W) -> (1, 1, C, T, H, W)
                input_tensor = np.stack(list(frames), axis=1)
                input_tensor = input_tensor[None, None]
                input_tensor = input_tensor.astype(np.float32)

                start_time = time.time()

                output = session.run(
                    [output_name],
                    {input_name: input_tensor}
                )[0]

                last_prediction_time = time.time() - start_time

                scores = output.reshape(-1)
                class_index = int(np.argmax(scores))

                # Если выход — логиты, вычисляем вероятности.
                # Если сумма уже близка к 1, считаем выход вероятностями.
                if (
                    np.any(scores < 0)
                    or not np.isclose(np.sum(scores), 1.0, atol=0.01)
                ):
                    shifted = scores - np.max(scores)
                    exp_scores = np.exp(shifted)
                    probabilities = exp_scores / np.sum(exp_scores)
                else:
                    probabilities = scores

                confidence = float(probabilities[class_index])
                current_word = str(classes[class_index])

                print(
                    "Результат:", current_word,
                    "| Оценка:", round(confidence * 100, 2), "%",
                    "| Время:", round(last_prediction_time, 2), "с"
                )

                # Следующее независимое окно из 32 кадров
                frames.clear()

            # Отображение результата
            display = frame.copy()

            cv2.rectangle(
                display,
                (0, 0),
                (display.shape[1], 85),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                display,
                "Распознано: " + current_word,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                "Оценка модели: " + str(round(confidence * 100, 1)) + "%",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            cv2.imshow("Распознавание РЖЯ", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
