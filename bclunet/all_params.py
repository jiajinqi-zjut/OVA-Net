

#TEST_DATA_PATH_5001500 = '//home//yxc//octfingerprint//bm3dimage//1310//'

NEW_TRAIN_DATA_PATH = 'SigTuple_data/New_Train_Data/'
TEST_DATA_PATH = 'SigTuple_data/Test_Data/'
TEST_DATA_PATH_SALT= 'SigTuple_data/Salt_Test_Data/'
SUBMISSION_DATA_PATH = 'SigTuple_data/Submission_Data/'
MODEL_CHECKPOINT_dir = '../3dunet_checkpoints_dataset/Checkpoints/noisy/'
log_dir='log_noisy/'
MODEL_CHECKPOINT_DIR_Test='Checkpoints_test10/'
#WEIGHTS = 'Model_Weights.hdf5'
AUGMENT_TRAIN_DATA = False
CREATE_EXTRA_DATA = True
IMG_ROWS = 240
IMG_COLS = 80
# IMG_ROWS = 400
# IMG_COLS = 1200

IMG_Z=7
TESTIMG_ROWS=496
TESTIMG_COLS=1800
TEST_N=21
IMG_START_NUM = 164
SMOOTH = 1.0
CLEAN_THRESH = 20
THRESH = 100
BATCH_SIZE = 8
EPOCHS = 3   # 300
BASE_LR = 1e-04
PATIENCE = 20

img_width = 1800
img_height = 500
# img_width = 900
# img_height = 500
img_num = 1400
#img_num=24
# img_num=700


img_width_houqin = 900
img_height_houqin = 500
# img_num_houqin=700
img_num_houqin=700