import json
import tensorflow as tf
# from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import sys
import os
import ast
from azureml.core.model import Model

sys.path.append("../models/research/")

# from object_detection.utils import visualization_utils as vis_util
# from object_detection.utils import label_map_util

def init():
    MODEL_NAME = 'object_detection'
    print("Tensorflow version: ", tf.__version__)
    model = Model.get_model_path(model_name=MODEL_NAME)

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
#     PATH_TO_FROZEN_GRAPH ='faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb'
    global detection_graph
    with tf.device('/cpu:0'):

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
def run_inference_for_single_image(image, graph):
    config = tf.ConfigProto(
            device_count = {'GPU': 0})

    with graph.as_default():
        with tf.Session(config=config) as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8).tolist()
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()
            output_dict['detection_scores'] = output_dict['detection_scores'][0].tolist()
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0].tolist()
    return output_dict
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
def base64ToImg(base64ImgString):

    if base64ImgString.startswith('b\''):

        base64ImgString = base64ImgString[2:-1]

    base64Img   =  base64ImgString.encode('utf-8')

    decoded_img = base64.b64decode(base64Img)

    img_buffer  = BytesIO(decoded_img)

    img = Image.open(img_buffer)
    return img

def run(raw_data):
    
    #     for image_path in TEST_IMAGE_PATHS:
    #   image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    print(raw_data)
    data = json.loads(raw_data)['data']
    image = base64ToImg(data)
#     print("Image shape", image.shape)
    image_np = load_image_into_numpy_array(image)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    print("output is: ", output_dict)
    return json.dumps(output_dict)

#     return output_dict
if __name__ == "__main__":
    print("Hello in main")
    PATH_TO_LABELS = os.path.join('../models/research/object_detection/data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90
#     label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#     categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)

    init()
    PATH_TO_TEST_IMAGES_DIR = '../models/research/object_detection/test_images'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (20, 20)
    with open(TEST_IMAGE_PATHS[0],"rb") as img:
        image_bytes = BytesIO(img.read())
    #image_data_1 = open(image_path_1, "rb").read()
    ENCODING = 'utf-8'
    # test_image = imresize(test_image,(width, height))
    payload = []
    encoded_image =base64.b64encode(image_bytes.getvalue())
    base64_string = encoded_image.decode(ENCODING)
# 
    image_request = {"data": "b'{0}'".format(base64_string)}
    # payload.append(image_request)
    # print(image_request)
    image_json = json.dumps(image_request)
#     print(image_json)

    output_dict = ast.literal_eval(run(image_json))
    
    print(output_dict)
#     image = Image.open(TEST_IMAGE_PATHS[0])

#     image_np = load_image_into_numpy_array(image)
#     vis_util.visualize_boxes_and_labels_on_image_array(
#     image_np,
#     np.array(output_dict['detection_boxes']),
#     np.array(output_dict['detection_classes']),
#     np.array(output_dict['detection_scores']),
#     category_index,
#     instance_masks=output_dict.get('detection_masks'),
#     use_normalized_coordinates=True,
#     line_thickness=8)
#     plt.figure(figsize=IMAGE_SIZE)
#     plt.imshow(image_np)