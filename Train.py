import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle
from network_class import Network

batch_size = 16
dataset_path = "./dataset/"
img_dim = [512, 512]
images_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
lr = 1e-4
epoch_number = 50000

images_path = []
masks_path = []
with open("Network", "rb") as file_:
    network = cPickle.load(file_)

with open(dataset_path + "train.txt", 'r') as file_:
    for l in file_:
        img, msk = l.strip('\n').split(' ')
        images_path.append(img)
        masks_path.append(msk)

images_path = tf.convert_to_tensor(images_path, dtype=tf.string)
masks_path = tf.convert_to_tensor(masks_path, dtype=tf.string)
queue = tf.train.slice_input_producer([images_path, masks_path], shuffle=False)
images = tf.image.decode_jpeg(tf.read_file(queue[0]), channels=3)
labels = tf.image.decode_png(tf.read_file(queue[1]), channels=1)

###Scaling
scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
h = tf.to_int32(tf.mul(tf.to_float(tf.shape(images)[0]), scale))
w = tf.to_int32(tf.mul(tf.to_float(tf.shape(images)[1]), scale))
coef = tf.squeeze(tf.pack([h, w]), squeeze_dims=[1])
images = tf.image.resize_images(images, coef)
labels = tf.image.resize_nearest_neighbor(tf.expand_dims(labels, 0), coef)
labels = tf.squeeze(labels, squeeze_dims=[0])
###Scaling

###Resize
images = tf.image.resize_image_with_crop_or_pad(images, img_dim[0], img_dim[1])
labels = tf.image.resize_image_with_crop_or_pad(labels, img_dim[0], img_dim[1])
###Resize

###RGB to BGR
img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=images)
images = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)
###RGB to BGR

images -= images_mean
image_batch, label_batch = tf.train.batch([images, labels], batch_size)
coordinator = tf.train.Coordinator()

###Network
nn = Network(network)
optm = tf.train.AdamOptimizer(learning_rate=lr)
trainable = tf.trainable_variables()
loss = nn.loss(image_batch, label_batch)
optimizer = optm.minimize(loss, var_list=trainable)
prediction = nn.preds(image_batch)
configuration = tf.ConfigProto()
session = tf.Session(config=configuration)
initialize = tf.initialize_all_variables()
session.run(initialize)
threads = tf.train.start_queue_runners(coord=coordinator, sess=session)

model_saver = tf.train.Saver(var_list=trainable, max_to_keep=40)

for epoch in range(epoch_number):
    loss_value, x = session.run([loss, optimizer])
    if (steps % 100) == 0:
        model_saver.save(session, "./checkpoint/", global_step=epoch)

coordinator.request_stop()
coordinator.join(threads)

