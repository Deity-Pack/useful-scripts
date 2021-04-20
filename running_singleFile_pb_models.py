import tensorflow as tf
import cv2
import numpy as np

GRAPH_PB_PATH = 'nyu2.pb'
input_data=cv2.resize(cv2.imread('../a/a.JPG'),(640,480))
input_data=input_data.astype(np.float32)
input_data=np.expand_dims(input_data,axis=0)

with tf.compat.v1.Session() as sess:
    with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)

    softmax_tensor = sess.graph.get_tensor_by_name('import/import/Identity:0') #output layer name
    predictions = sess.run(softmax_tensor, {'import/x:0': input_data}) #input layer name
    print(predictions.shape)
    predictions= np.squeeze(predictions, axis=0)
    print(predictions.shape)
    predictions = predictions.astype(np.uint8)
    cv2.imshow('final',predictions)
    cv2.waitKey(0)