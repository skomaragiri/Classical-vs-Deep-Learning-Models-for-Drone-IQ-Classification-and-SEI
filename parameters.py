# DERIVATION PARAMETERS
NUM_FEATURES = 7
WINDOW_LEN = 1024 # measured in IQ samples
FFT_OVERLAP = 0.1 # measured as a fraction of the WINDOW_LEN

PATCH_WIDTH_TIME = 16
PATCH_HEIGHT_FREQ = 16
TIME_HOP = 8
FREQ_HOP = 8


# DATA PARAMETERS
# NUM_TRAINING_FILES = 28 # how many files in the saved numpy data for training
# NUM_EVALUATION_FILES = 6 # how many files in the saved numpy data for evaluation
NUM_TRAINING_FILES = 100 # how many files in the saved numpy data for training
NUM_EVALUATION_FILES = 100 # how many files in the saved numpy data for evaluation
MAX_FILES = 100 # if not using the saved numpy data, this is the max files to intake
USE_SAVED_DATA = True # True = used the saved .npy file data instead of re-deriving the features again
SAVE_METRICS_TO_FILE = True

TRAINING_DATASET = ""
EVAL_DATASET = ""

FEATURES_TO_USE=''
TRAINING_DATASET = "n11_pro_comms_room_DL"
EVAL_DATASET = "n11_pro_comms_room_DL"

TRAINING_DATASET = "dji_10mhz_chamber"
EVAL_DATASET = "dji_10mhz_chamber"

# A very noisy dataset that usually has terrible results
TRAINING_DATASET = "dji_20mhz_comms"
EVAL_DATASET = "dji_20mhz_comms"

# Post fixes 
PREPROCESSING_TECHNIQUE = f"fft_w{PATCH_WIDTH_TIME}_h{PATCH_HEIGHT_FREQ}_th{TIME_HOP}_fh{FREQ_HOP}"
# PREPROCESSING_TECHNIQUE = ""


# DATASET PARAMETERS
# carlos' Mac
# data_dir = '/Users/carlos_1/Documents/GitHub/RFML-Code/RFML_Combined_Dataset_2025/RFML_Drone_Dataset_2025/old_drone_full_annotated_dataset/RFML_Old_Drone_Eval_data/*'
# uav-cyberlab-rfml laptop
training_data_dir = f'/home/uav-cyberlab-rfml/RFML/test-dataset/test_{TRAINING_DATASET}_training'
eval_data_dir = f'/home/uav-cyberlab-rfml/RFML/test-dataset/test_{TRAINING_DATASET}_eval'
deep_training_dir = f'/home/uav-cyberlab-rfml/rww_conference/annotations/Training'


# FEATURES_TO_USE = [
#     # TODO: FIGURE OUT HOW TO MAKE THE RETURN FUNCTION ACTUALLY UPDATE
#         # checkout preprocessing.py: extract_rf_features() and try to tie that function to this
#     # "mean_I",
#     # "mean_Q",
#     "var_I",
#     "var_Q",
#     # "skew_I",
#     # "kurt_I",
#     # "skew_Q",
#     # "kurt_Q",
#     "mean_abs",
#     # "var_abs",
#     # "PAPR",
#     # "corr_IQ",
#     # "subbands",
#     "total_power",
#     "spectral_centroid", # you need centroid with bandwidth
#     "spectral_bandwidth", # you need centroid with bandwidth
#     # "spectral_flatness",
# ]