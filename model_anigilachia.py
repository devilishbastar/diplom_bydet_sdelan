import onnx

model = onnx.load("S3D.onnx")

print("Входы:")
for inp in model.graph.input:
    print(inp.name, inp.type.tensor_type.shape)

print("\nВыходы:")
for out in model.graph.output:
    print(out.name, out.type.tensor_type.shape)

print("\nПоследние операции:")
for node in model.graph.node[-15:]:
    print(node.op_type, node.name, node.output)