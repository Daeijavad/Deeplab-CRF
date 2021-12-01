import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle
from network_class import Network, CRFlayer

dataset_path = "./dataset/"
images_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
images_path = []
masks_path = []
test_case_number = 1449
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

###RGB to BGR
img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=images)
images = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)
###RGB to BGR

images -= images_mean
image_batch, label_batch = tf.expand_dims(images, dim=0), tf.expand_dims(labels, dim=0)
coordinator = tf.train.Coordinator()

###Network
nn = Network(network)
trainable = tf.trainable_variables()
prediction = nn.preds(image_batch)
mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, label_batch, num_classes=21)
m = tf.keras.metrics.MeanIoU(num_classes=21)
configuration = tf.ConfigProto()
session = tf.Session(config=configuration)
initialize = tf.initialize_all_variables()
session.run(initialize)
sess.run(tf.initialize_local_variables())
saver = tf.train.Saver(var_list=trainable)
saver.restore(session, './checkpoint/model.ckpt')
model_saver = tf.train.Saver(var_list=trainable, max_to_keep=40)

threads = tf.train.start_queue_runners(coord=coordintor, sess=session)

for test_case in range(test_case_number):
    preds, _ = session.run([prediction, update_op])
    mask = np.array(preds)[0, 0, :, :, 0]
    mask_img = Image.new('RGB', (len(mask[0]), len(mask)))
    p = mask_img.load()
    for x, j in enumerate(mask):
        for y, k in enumerate(j):
            if k < 21:
                p[y,x] = color_map[k]
    mask_img = Image.fromarray(np.array(mask_img))
    i = cv2.imread(images_path[test_case])
    crf_output = CRFlayer(i, mask_img)
    m.update_state(crf_output, cv2.imread(images_path[test_case][0:-4] + '.png'))

print('Mean IoU: {:.3f}'.format(mIoU.eval(session=session)))
coordinator.request_stop()
coordinator.join(threads)
print(m.result().numpy())
