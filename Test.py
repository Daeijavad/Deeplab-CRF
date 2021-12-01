import tensorflow as tf
import numpy as np
from PIL import Image
from network_class import Network, CRFlayer
from six.moves import cPickle
import cv2


image_path="./Test_Cases/9.jpg"
weight_path = "./checkpoint/model.ckpt"
with open("./Network", "rb") as f:
    network = cPickle.load(f)
image_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
color_map = [(0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),
             (0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0),(192,128,0),
             (64,0,128),(192,0,128),(64,128,128),(192,128,128),(0,64,0),(128,64,0),
             (0,192,0),(128,192,0),(0,64,128)]

nn = Network(network)
trainable = tf.trainable_variables()
saver = tf.train.Saver(var_list=trainable)
    
image = tf.image.decode_jpeg(tf.read_file(image_path), channels=3)
r, g, b = tf.split(value=image, num_or_size_splits=3, axis=2)
image = tf.cast(tf.concat(axis=2, values=[b, g, r]), dtype=tf.float32)
image -= image_mean

prediction = nn.preds(tf.expand_dims(image, dim=0))
configuration = tf.ConfigProto()
session = tf.Session(config=configuration)
initialize = tf.initialize_all_variables()
session.run(initialize)
saver.restore(session, weight_path)
infer = session.run([prediction])
mask = np.array(infer)[0, 0, :, :, 0]
mask_img = Image.new('RGB', (len(mask[0]), len(mask)))
p = mask_img.load()
for x, j in enumerate(mask):
    for y, k in enumerate(j):
        if k < 21:
            p[y,x] = color_map[k]
mask_img = Image.fromarray(np.array(mask_img))
mask_img.save(image_path[0:-4] + '_mask.png')

i = cv2.imread(image_path)
m = cv2.imread(image_path[0:-4] + '_mask.png')
crf_output = CRFlayer(i, m)
cv2.imwrite(image_path[0:-4] + "_CRFmask.png", crf_output)
