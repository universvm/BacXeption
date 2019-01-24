import os

from datetime import datetime

# Project:
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_NAME = 'bacxeption'
START_DATE = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
TRAIN_MODE = True

# Images:
FORMAT_IMG = ('.tiff', '.tif')
INPUT_DIM = (180, 180, 3)
MIN_PX_AREA = 50
EXTRA_BORDER_PX = 5

# Network:
BATCH_SIZE = 10
EPOCHS = 300
LOSS_FUNC = 'mean_squared_error'
METRICS = ['accuracy']
N_CLASSES = 2
OPTIMIZER = 'adam'
TRAIN_IMG_PATH = os.path.join(ROOT_DIR, PROJECT_NAME, 'data')

# Logger:
if TRAIN_MODE:
    LOG_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'models', START_DATE)
else:
    LOG_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'models')
os.makedirs(LOG_DIR, exist_ok=True)


# Test:
TEST_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'to_predict')
OUTPUT_COORDS = True
OUTPUT_GRAPH = True
