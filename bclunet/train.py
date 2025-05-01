import argparse
import os

from keras.callbacks import ModelCheckpoint, TensorBoard

from data_load import train_generator
from model import model_3dunet_res_lstm_sweat


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='use which gpu', default=0, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='where to save the checkpoint',
                        default='train', type=str)

    parser.add_argument('--resume', dest='resume',
                        help='resume or not',
                        default=False, type=bool)
    parser.add_argument('--epoch', dest='epoch',
                        help='load which checkpoint',
                        default=0, type=int)

    args = parser.parse_args()
    return args


args = parse_args()
print(f"=====-----> args={args}")
os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu_id}'

epochs = 100
train_batch_size = 2
train_size = 11928 // 5
save_dir = f"./models/{args.save_dir}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_generators = train_generator("./train.tfrecords", train_batch_size)

checkpointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.h5'),
                               verbose=1, save_weights_only=False, period=1)
tensorboard = TensorBoard('./logs/', write_graph=True, update_freq='batch')
callbacks = [checkpointer, tensorboard]

model = model_3dunet_res_lstm_sweat()
model.summary()

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generators,
                    steps_per_epoch=(train_size // train_batch_size),
                    epochs=epochs,
                    callbacks=callbacks,
                    max_queue_size=2)
