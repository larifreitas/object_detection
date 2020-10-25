import os
import cv2
import tarfile
import pathlib
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib
import tensorflow.compat.v1 as tf_v1  # para compatibilidade com session

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

tf.compat.v1.disable_resource_variables()

cap = cv2.VideoCapture(0)

# models garden
if 'models' in pathlib.Path.cwd().parts:
    while 'models' in pathlib.Path.cwd().parts:
        os.chdir('..')
elif not pathlib.Path('models').exists():
    os.system('git clone --depth 1 https://github.com/tensorflow/models')

# path model
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
model_file = model_name + '.tar.gz'
download_base = 'http://download.tensorflow.org/models/object_detection/'

# path frozen graph
path_detection_frozen = model_name + '/frozen_inference_graph.pb'

# path label
path_to_labels = os.path.join('models/research/object_detection/data/mscoco_label_map.pbtxt')

# num classes 1 at 90
num_class = 1

# download model
opener = urllib.request.URLopener().retrieve(download_base + model_file, model_file)
tar_file = tarfile.open(model_file)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# compat
os.system('''tf_upgrade_v2 \
  --intree my_project/ \
  --outtree my_project_v2/''')

# load frozen model
# importando modelo congelado ao script para carregar modelo
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(path_detection_frozen, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# label
# os rótulos de mscoco_label_map em https://github.com/tensorflow/models/tree/master/research/object_detection/data
label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_class, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# compat Session
# compatibilidade com Session
tf_v1.disable_v2_behavior()

# lendo gráfico para detecção
with detection_graph.as_default():
    with tf_v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # expandir as dimensões, pois o modelo espera que as imagens tenham forma
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            #  cada caixa representa uma parte da imagem, tensor de caixas de detecção
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # tensor de pontuação de detecção
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # correspondem as chaves no mapa de rótulos, tensor de classes de detecção
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # sessão pra previsão
            (boxes, classes, scores, detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )
            # labels e boxes
            vis_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(scores).astype(np.uint32),
                np.squeeze(classes),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4
            )
            # exibição
            cv2.imshow('object detection person', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
