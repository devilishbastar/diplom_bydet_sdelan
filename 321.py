import onnxruntime as ort

session = ort.InferenceSession("SignFlow-R.onnx")
meta = session.get_modelmeta()

print("custom_metadata_map:", meta.custom_metadata_map)
print("description:", meta.description)
print("domain:", meta.domain)
print("graph_name:", meta.graph_name)
print("producer_name:", meta.producer_name)
print("version:", meta.version)