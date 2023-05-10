import tensorflow as tf
import config

def _parse_function(filename, label):
    filename = config.DATA_DIR + '/' + filename
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_height = tf.shape(image_decoded)[0]
    image_width = tf.shape(image_decoded)[1]
    random_s = tf.random_uniform([1], minval=256, maxval=481, dtype=tf.int32)[0]
    resized_height, resized_width = tf.cond(image_height < image_width,
                                            lambda: (random_s, tf.cast(tf.multiply(tf.cast(image_width, tf.float64), tf.divide(random_s, image_height)), tf.int32)),
                                            lambda: (tf.cast(tf.multiply(tf.cast(image_height, tf.float64), tf.divide(random_s, image_width)), tf.int32), random_s))
    image_float = tf.image.convert_image_dtype(image_decoded, float32)
    image_resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    image_cropped = tf.random_crop(image_resized, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL])
    image_flipped = tf.image.random_flip_left_right(image_cropped)
    image_distorted = tf.image.random_brightness(image_flipped, max_delta=60)
    image_distorted = tf.image.random_contrast(image_distorted, lower=0.2, upper=2.0)
    image_distorted = tf.image.per_image_standardization(image_distorted)
    onehot_label = tf.one_hot(label, config.CLASS_NUM)
    return image_distorted, onehot_label

def make_data(train_file, batch_size):
    record_defaults = [tf.string, tf.int32]
    dataset = tf.contrib.data.CsvDataset(train_file, record_defaults)
    dataset = dataset.shuffle(buffersize=10000000)
    dataset = dataset.map(_parse_function, num_parallel_call=4)
    dataset = dataset.repeat(10)
    dataset.batch(batch_size)
    dataset.prefetch(batch_size)