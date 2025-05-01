import tensorflow as tf


def parse_data_train(tfrecord_file):
    features = {
        "image": tf.FixedLenFeature([], tf.string, default_value=""),
        "line": tf.FixedLenFeature([], tf.string, default_value=""),
        "truth": tf.FixedLenFeature([], tf.string, default_value=""),
    }
    parsed = tf.io.parse_single_example(tfrecord_file, features)
    image = tf.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, [480, 128, 1])
    truth = tf.decode_raw(parsed['truth'], tf.float32)
    truth = tf.reshape(truth, [480, 128, 1])
    images = tf.cast(image, tf.float32)
    truths = tf.cast(truth, tf.int32)
    print("---------------------image---------------------", images)
    print("---------------------truth---------------------", truths)
    images = images / 255.0
    return images, truths


def parse_data_test(tfrecord_file):
    features = {
        "image": tf.FixedLenFeature([], tf.string, default_value=""),
        "truth": tf.FixedLenFeature([], tf.string, default_value=""),
    }
    parsed = tf.io.parse_single_example(tfrecord_file, features)
    image = tf.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, [480, 1792, 1])
    truth = tf.decode_raw(parsed['truth'], tf.float32)
    truth = tf.reshape(truth, [480, 1792, 1])
    images = tf.cast(image, tf.float32)
    truths = tf.cast(truth, tf.int32)
    # print("---------------------image---------------------",images)
    # print("---------------------bound---------------------",bounds)
    # print("---------------------truth---------------------",truths)
    images = images / 255.0
    return images, truths


def train_generator(tf_data, batchsize, shuffle=True):
    '''
    Creates a python generator that loads the AVA dataset images with random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.TFRecordDataset(tf_data)
        train_dataset = train_dataset.map(parse_data_train, num_parallel_calls=2)
        train_dataset = train_dataset.batch(5, drop_remainder=True)
        train_dataset = train_dataset.shuffle(buffer_size=50)
        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=50)
        train_iterator = train_dataset.make_initializable_iterator()
        train_batch = train_iterator.get_next()
        sess.run(train_iterator.initializer)
        while True:
            try:
                x_batch, y_batch = sess.run(train_batch)
                yield (x_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                x_batch, y_batch = sess.run(train_batch)
                yield (x_batch, y_batch)


def val_generator(tf_data, batchsize):
    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for validation set

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        val_dataset = tf.data.TFRecordDataset(tf_data)
        val_dataset = val_dataset.map(parse_data_test)
        train_dataset = train_dataset.batch(1)
        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()
        val_batch = val_iterator.get_next()
        sess.run(val_iterator.initializer)
        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
