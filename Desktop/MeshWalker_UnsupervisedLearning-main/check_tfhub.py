import os

import tensorflow as tf
import tensorflow_hub as hub

os.environ["TFHUB_CACHE_DIR"] = os.path.expanduser('~') + '/tfhub-modules-cache/'

m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/tensorflow/resnet_50/classification/1")
])
m.build([None, 224, 224, 3])

