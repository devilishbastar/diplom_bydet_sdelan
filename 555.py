import cv2
import onnxruntime as ort

MODEL_PATH = "D:\diplom\SignFlow-R.onnx"


session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("\n=== ВХОДЫ МОДЕЛИ ===")
for item in session.get_inputs():
    print(
        "Имя:", item.name,
        "| Форма:", item.shape,
        "| Тип:", item.type
    )

print("\n=== ВЫХОДЫ МОДЕЛИ ===")
for item in session.get_outputs():
    print(
        "Имя:", item.name,
        "| Форма:", item.shape,
        "| Тип:", item.type
    )


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Не удалось открыть веб-камеру"
    )

print("\nНажми Q для выхода.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр")
        break

    cv2.imshow("SingFlow-R - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()