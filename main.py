from PIL import Image, ImageDraw
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="face_detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img = Image.open("test.png").resize((640, 640)).convert("RGB")
input_data = np.expand_dims(np.array(img)/255, axis=0).astype("float32")
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

output_data_transposed=output_data[0].T
def draw_bbox_on_image(image,x, y, w, h):
    # Denormalize the coordinate
    
    x = int(x * image.shape[1])
    y = int(y * image.shape[0])
    w = int(w * image.shape[1])
    h = int(h * image.shape[0])

    img=Image.fromarray(image)
    draw=ImageDraw.Draw(img)
    draw.rectangle([(x-w/2, y-h/2), (x+w/2,y+h/2)],outline= (0, 255, 0), width=2)
    return np.array(img)
    
sorted_indices = np.argsort(output_data_transposed[:, 4])[::-1]

top_K_by_confidence = output_data_transposed[sorted_indices[0:10]]
print(top_K_by_confidence)
for bbox in top_K_by_confidence:
    xywh = bbox[:4]
    img = draw_bbox_on_image(np.array(img), xywh[0], xywh[1], xywh[2], xywh[3])
Image.fromarray(img).save("result.png")
