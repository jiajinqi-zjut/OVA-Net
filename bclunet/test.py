import argparse
import os

import cupy as cp
import keras.backend.tensorflow_backend as K
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

from model import model_3dunet_res_lstm_sweat


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='use which gpu', default=0, type=int)

    parser.add_argument('--cp_dir', dest='cp_dir',
                        help='directory to save models',
                        default=None)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save data',
                        default=None)
    parser.add_argument('--epoch', dest='epoch',
                        help='load which checkpoint',
                        default=1, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='load which checkpoint',
                        default=1, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch',
                        help='load which checkpoint',
                        default=1, type=int)
    return parser.parse_args()


def HD_AMPD_RASMPD(a, b):
    # a = a.squeeze(0)
    # b = b.squeeze(0)
    min_a = []
    min_a2 = []
    min_b = []
    min_b2 = []
    for i in range(len(a)):
        min = np.sqrt(np.square(a[i] - b[0]) + np.square(i - 0))
        for j in range(len(a)):
            temp = np.sqrt(np.square(a[i] - b[j]) + np.square(i - j))
            if min > temp:
                min = temp
        min_a.append(min)
        min_a2.append(np.square(min))
    for i in range(len(a)):
        min = np.sqrt(np.square(b[i] - a[0]) + np.square(i - 0))
        for j in range(len(a)):
            temp = np.sqrt(np.square(b[i] - a[j]) + np.square(i - j))
            if min > temp:
                min = temp
        min_b.append(min)
        min_b2.append(np.square(min))
    print("max_a---------->", np.max(min_a))
    print("max_b---------->", np.max(min_b))
    return np.max([np.max(min_a), np.max(min_b)]), (np.sum(min_a) + np.sum(min_b)) / (2 * len(a)), np.sqrt(
        (np.sum(min_a2) + np.sum(min_b2)) / (2 * len(a)))


def HD_AMPD_RASMPD_cupy(a, b):
    with cp.cuda.Device(0):
        a = cp.asarray(a)
        b = cp.array(b)

        min_a = cp.sqrt(
            cp.square(a[:, cp.newaxis] - b) + cp.square(cp.arange(len(a))[:, cp.newaxis] - cp.arange(len(b))))
        min_a = cp.min(min_a, axis=1)
        min_a2 = cp.square(min_a)

        min_b = cp.sqrt(
            cp.square(b[:, cp.newaxis] - a) + cp.square(cp.arange(len(b))[:, cp.newaxis] - cp.arange(len(a))))
        min_b = cp.min(min_b, axis=1)
        min_b2 = cp.square(min_b)

        max_value = cp.max(cp.concatenate([min_a, min_b]))
        avg_value = (cp.sum(min_a) + cp.sum(min_b)) / (2 * len(a))
        std_value = cp.sqrt((cp.sum(min_a2) + cp.sum(min_b2)) / (2 * len(a)))

        return cp.asnumpy(max_value), cp.asnumpy(avg_value), cp.asnumpy(std_value)


def _parse_read(tfrecord_file):
    features = {
        "image": tf.FixedLenFeature([], tf.string, default_value=""),
        "line": tf.FixedLenFeature([], tf.string, default_value=""),
        "truth": tf.FixedLenFeature([], tf.string, default_value=""),
    }
    parsed = tf.io.parse_single_example(tfrecord_file, features)
    image = tf.decode_raw(parsed['image'], tf.float32)
    image = tf.reshape(image, [480, 1792, 1])
    truth = tf.decode_raw(parsed['truth'], tf.float32)
    truth = tf.reshape(truth, [480, 1792])
    images = tf.cast(image, tf.float32)
    truths = tf.cast(truth, tf.int32)
    # print("---------------------image---------------------",images)
    # print("---------------------bound---------------------",bounds)
    # print("---------------------truth---------------------",truths)
    images = images / 255.0
    return images, truths


def collect(label_pre):
    label0 = []
    label1 = []
    label2 = []
    h, w = label_pre.shape
    for i in range(w):
        tf0 = False
        tf1 = False
        tf2 = False
        for j in range(1, h - 1):
            if label_pre[j - 1, i] == 0 and label_pre[j, i] == 1 and tf0 == False:
                label0.append(j)
                tf0 = True
            if label_pre[j, i] == 1 and label_pre[j + 1, i] == 2 and tf1 == False:
                label1.append(j)
                tf1 = True
            if label_pre[j, i] == 2 and label_pre[j + 1, i] == 0 and tf2 == False:
                label2.append(j)
                tf2 = True
        if tf0 == False:
            if len(label0) == 0:
                label0.append(0)
            else:
                label0.append(label0[len(label0) - 1])
        if tf1 == False:
            if len(label1) == 0:
                label1.append(0)
            else:
                label1.append(label1[len(label1) - 1])
        if tf2 == False:
            if len(label2) == 0:
                label2.append(0)
            else:
                label2.append(label2[len(label2) - 1])
    return label0, label1, label2


def collect_own(label):
    label0 = np.argmax(label == 1, axis=0)
    label1 = np.argmax(label == 2, axis=0) - 1
    # 将整个ndarray上下翻转，找到反转后的第一个==2的像素，然后返回值=高度减去翻转后的index
    label2 = (label.shape[0] - 1) - np.argmax((label[::-1, :]) == 2, axis=0)
    return label0, label1, label2


def test_mae(args, model_save_dir):
    dataset = tf.data.TFRecordDataset("./val.tfrecords")
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(mini_batch_size, drop_remainder=True)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    images_batch, truths_batch = iterator.get_next()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    with tf.device("/gpu:%d" % args.gpu_id):
        for epoch in range(args.start_epoch, args.end_epoch + 1):
            # 创建csv文件夹和文件
            epoch_str = "{:03d}".format(epoch)
            dir_path = f'{path}/results/{args.save_dir}'
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            csv_file = f'{path}/results/{args.save_dir}/{args.cp_dir}-{epoch_str}.csv'
            if not os.path.exists(csv_file):
                with open(csv_file, 'w') as file:
                    file.write(
                        'train,checkpoint,image,miou1,miou2,'
                        'layer1_MAE,layer1_MSE,HDS1,AMPDS1,RASMPDS1,'
                        'layer2_MAE,layer2_MSE,HDS2,AMPDS2,RASMPDS2,'
                        'layer3_MAE,layer3_MSE,HDS3,AMPDS3,RASMPDS3\n')

            image = tf.placeholder(tf.float32, shape=[batch_size, mini_batch_size, height, width, 1], name="image")
            truth = tf.placeholder(tf.int32, shape=[batch_size, mini_batch_size, height, width], name="truth")
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)

            # load model
            model = model_3dunet_res_lstm_sweat(False)
            sess.run(tf.global_variables_initializer())
            epoch_str = "{:03d}".format(epoch)
            model.load_weights(f"{model_save_dir}/model_{epoch_str}.h5")
            logit = model(image)

            with tqdm(initial=0, total=int(num_in_epoch), desc=f"checkpoint {epoch_str}") as pbar:
                try:
                    # num_in_epoch = len(dataset) / batch_size
                    for step in range(num_in_epoch):
                        image_in, truths_in = sess.run([images_batch, truths_batch])
                        out = sess.run([logit], feed_dict={image: image_in, truth: truths_in})
                        out = np.argmax(out, axis=-1)
                        out = np.squeeze(out)
                        label = np.squeeze(truths_in)

                        iou = miou(generate_matrix(5, label, out))
                        print_to_csv_file(label[0], out[0], csv_file, epoch, iou)
                        print_to_csv_file(label[1], out[1], csv_file, epoch, iou)
                        print_to_csv_file(label[2], out[2], csv_file, epoch, iou)
                        print_to_csv_file(label[3], out[3], csv_file, epoch, iou)
                        print_to_csv_file(label[4], out[4], csv_file, epoch, iou)

                        pbar.update(1)
                except tf.errors.OutOfRangeError:
                    print('Epoch limit reached')
                finally:
                    sess.close()


def compute_miou(label, out):
    label = label[:, 200:1600]
    out = out[:, 200:1600]

    and_sum_1 = np.sum(np.logical_and(np.where(label == 1, True, False), np.where(out == 1, True, False)))
    or_sum_1 = np.sum(np.logical_or(np.where(label == 1, True, False), np.where(out == 1, True, False)))
    miou1 = and_sum_1 / or_sum_1
    and_sum_2 = np.sum(np.logical_and(np.where(label == 2, True, False), np.where(out == 2, True, False)))
    or_sum_2 = np.sum(np.logical_or(np.where(label == 2, True, False), np.where(out == 2, True, False)))
    miou2 = and_sum_2 / or_sum_2
    return miou1, miou2


def generate_matrix(num_class, gt_image, pre_image):
    # 正确的gt_mask
    mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask

    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)  # (n, n)
    return confusion_matrix


def miou(hist):
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    return iou


def print_to_csv_file(label, out, csv_file, epoch, iou):
    label1, label2, label3 = collect_own(label)
    label_pre1, label_pre2, label_pre3 = collect_own(out)

    mae1 = torch.mean(
        torch.abs(torch.tensor(label_pre1, dtype=torch.float32) - torch.tensor(label1, dtype=torch.float32)))
    mae2 = torch.mean(
        torch.abs(torch.tensor(label_pre2, dtype=torch.float32) - torch.tensor(label2, dtype=torch.float32)))
    mae3 = torch.mean(
        torch.abs(torch.tensor(label_pre3, dtype=torch.float32) - torch.tensor(label3, dtype=torch.float32)))

    layer1_MSE = np.sum(
        np.power([value1 - value2 for value1, value2 in zip(label1[200:1600], label_pre1[200:1600])], 2)) / 1400
    layer2_MSE = np.sum(
        np.power([value1 - value2 for value1, value2 in zip(label2[200:1600], label_pre2[200:1600])], 2)) / 1400
    layer3_MSE = np.sum(
        np.power([value1 - value2 for value1, value2 in zip(label3[200:1600], label_pre3[200:1600])], 2)) / 1400

    HDS1, AMPDS1, RASMPDS1 = HD_AMPD_RASMPD_cupy(label1[200:1600], label_pre1[200:1600])
    HDS2, AMPDS2, RASMPDS2 = HD_AMPD_RASMPD_cupy(label2[200:1600], label_pre2[200:1600])
    HDS3, AMPDS3, RASMPDS3 = HD_AMPD_RASMPD_cupy(label3[200:1600], label_pre3[200:1600])

    # miou1, miou2 = compute_miou(label, out)
    miou1, miou2 = iou[1], iou[2]
    with open(csv_file, 'a') as file:
        print(
            f'{args.cp_dir},{epoch},None,{miou1},{miou2},{mae1},{layer1_MSE},{HDS1},{AMPDS1},{RASMPDS1},'
            f'{mae2},{layer2_MSE},{HDS2},{AMPDS2},{RASMPDS2},{mae3},{layer3_MSE},{HDS3},{AMPDS3},{RASMPDS3}',
            file=file)


if __name__ == '__main__':
    args = parse_args()
    print(f"=====-----> args={args}")
    image_num = 200
    batch_size = 1
    mini_batch_size = 5
    num_in_epoch = int(image_num / mini_batch_size)
    height = 480
    width = 1792

    path = './'
    model_save_dir = f'./models/{args.cp_dir}'
    test_mae(args, model_save_dir)
