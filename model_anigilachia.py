import onnx
from onnx import numpy_helper

model = onnx.load("s3d.onnx")

# Ищем последний классификационный Conv
for node in model.graph.node:
    if node.name == "/model/classifier/classifier.1/Conv":
        print("Найден классификатор!")
        print("Входы:", list(node.input))
        print("Выходы:", list(node.output))

        # Проверяем веса и смещения
        for name in node.input:
            for initializer in model.graph.initializer:
                if initializer.name == name:
                    weights = numpy_helper.to_array(initializer)
                    print(
                        "Параметр:", name,
                        "Размерность:", weights.shape
                    )
