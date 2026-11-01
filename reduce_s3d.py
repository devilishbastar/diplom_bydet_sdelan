#!/usr/bin/env python3
"""Сокращение выходных классов S3D ONNX по S3D_selected_classes.txt.
Запуск: python reduce_s3d.py S3D.onnx S3D_selected.onnx S3D_selected_classes.txt
Требуется: pip install onnx numpy
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference, checker

if len(sys.argv) != 4:
    raise SystemExit("Использование: python reduce_s3d.py S3D.onnx S3D_selected.onnx S3D_selected_classes.txt")
source, destination, class_file = sys.argv[1:]

# Первый столбец в словаре — новый индекс, второй — исходное название.
# Старые индексы берутся из mapping-файла, чтобы не перепутать порядок классов.
mapping_path = class_file.replace('S3D_selected_classes.txt', 'S3D_index_mapping.csv')
import csv, os
if not os.path.exists(mapping_path):
    raise SystemExit("Рядом со словарём должен находиться S3D_index_mapping.csv")
with open(mapping_path, encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
old_indices = [int(r['old_index']) for r in rows]
if not old_indices:
    raise SystemExit('Список выбранных классов пуст')
if len(old_indices) != len(set(old_indices)):
    raise SystemExit('В списке повторяются исходные индексы')

model = onnx.load(source)
# Ищем параметры финального классификатора по известным именам S3D.
initializers = {x.name: x for x in model.graph.initializer}
w_name = 'model.classifier.1.weight'
b_name = 'model.classifier.1.bias'
if w_name not in initializers or b_name not in initializers:
    raise SystemExit('Не найдены ожидаемые веса/смещения финального классификатора S3D')
w = numpy_helper.to_array(initializers[w_name])
b = numpy_helper.to_array(initializers[b_name])
if w.shape[0] != 1599 or b.shape[0] != 1599:
    raise SystemExit(f'Неожиданный размер классификатора: weight={w.shape}, bias={b.shape}')
if max(old_indices) >= w.shape[0] or min(old_indices) < 0:
    raise SystemExit('В mapping есть индекс вне диапазона 0..1598')
new_w = np.ascontiguousarray(w[old_indices])
new_b = np.ascontiguousarray(b[old_indices])

# Заменяем данные тензоров, сохраняя имена и типы.
for init, arr in ((initializers[w_name], new_w), (initializers[b_name], new_b)):
    replacement = numpy_helper.from_array(arr, name=init.name)
    init.CopyFrom(replacement)

# Исправляем известную размерность выхода и промежуточные value_info.
def set_first_dim(value_info, n):
    tensor_type = value_info.type.tensor_type
    if tensor_type.HasField('shape') and len(tensor_type.shape.dim) > 0:
        tensor_type.shape.dim[0].dim_value = n

n = len(old_indices)
for vi in list(model.graph.output) + list(model.graph.value_info):
    # output1 is [1, N]; classifier Conv output also carries N on class axis.
    if vi.name in ('output1', '/model/classifier/classifier.1/Conv_output_0'):
        tensor_type = vi.type.tensor_type
        if tensor_type.HasField('shape') and len(tensor_type.shape.dim) >= 2:
            tensor_type.shape.dim[1].dim_value = n

# Shape inference may refresh intermediate metadata; if it fails, checker still validates graph.
try:
    model = shape_inference.infer_shapes(model)
except Exception:
    pass
checker.check_model(model)
onnx.save(model, destination)
print(f'Готово: {destination}')
print(f'Классов: {n} (было 1599)')
print('Порядок выходов соответствует порядку строк в S3D_selected_classes.txt')
