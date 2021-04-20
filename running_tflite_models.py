import tensorflow as tf
import numpy as np
import cv2
interpreter = tf.lite.Interpreter(model_path="nyu2tflite.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details,output_details)

# input_data=cv2.resize(cv2.imread('../a/a.JPG'),(640,480))
# inp = input_data
# input_data=input_data.astype(np.float32)
# input_data=np.expand_dims(input_data,axis=0)
# print(input_data.shape)
#
#
#
# interpreter.set_tensor(input_details[0]['index'], input_data)
# interpreter.invoke()
#
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data.shape)
# output_data = np.squeeze(output_data,axis=0)
# print(output_data.shape)
#
# output_data = output_data.astype(np.uint8)
# print(output_data)
#
# cv2.imshow('final',output_data)
# cv2.imshow('bdsfbjdsbfkj',inp)
# cv2.waitKey(0)

