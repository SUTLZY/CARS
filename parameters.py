BATCH_SIZE = 64 #8
INPUT_DIM = 4
EMBEDDING_DIM = 128
SAMPLE_SIZE = 200
K_SIZE = 20
BUDGET_RANGE = (6, 8)
SAMPLE_LENGTH = 0.2

ADAPTIVE_AREA = True
ADAPTIVE_TH = 0.4

USE_GPU = False
USE_GPU_GLOBAL = True
#CUDA_DEVICE = [0, 1, 2, 3]
CUDA_DEVICE = [1]
#NUM_META_AGENT = 32 #6
NUM_META_AGENT = 12
LR = 1e-4
GAMMA = 1
DECAY_STEP = 32
SUMMARY_WINDOW = 8
#FOLDER_NAME = 'ipp-vae'  # 001是ae使用next_ob  002无ae
##FOLDER_NAME = 'test-01'
#model_path = f'data/model/{FOLDER_NAME}'
#train_path = f'data/train/{FOLDER_NAME}'
#gifs_path = f'data/gifs/{FOLDER_NAME}'


RANDOM_ENV = True
USE_VAE = False # True
# 使用下一时刻观测做loss增加的参数
NEXT_OB = True
USE_WAE = True
Aug_S   = False
USE_VAE_A = False #True
MMD_LOSS_COEF = 100
PREDICT_LOSS_COEF = 0.1
TOTAL_WAE_LOSS_COEF = 1 # 0.5 和 1 reward上看区别不大  各种loss、remain Budget有变化
LOAD_MODEL = True
SEED = 369

MY_VAR = "ipp"
FOLDER_NAME = "2024-11-20T19:44:49 | ipp | "
FOLDER_LOGO = FOLDER_NAME + f"RANDOM_ENV:{RANDOM_ENV}" + f" |  VAE:{USE_VAE_A} | Forward:{USE_VAE} | WAE:{USE_WAE} | AUG_S:{Aug_S}"
#FOLDER_LOGO = FOLDER_NAME + f"RANDOM_ENV:{RANDOM_ENV}" + f" | VAE:{USE_VAE_A}"

model_path = f'../data/model/{FOLDER_LOGO}'
train_path = f'../data/train/{FOLDER_LOGO}'
gifs_path = f'../data/gifs/{FOLDER_LOGO}'


SAVE_IMG_GAP = 1000
LEN_HISTORY = 5
HIDDEN_DIM = 32 #200
#LATENT_DIM = 200
LATENT_DIM = 16 #128
STEP_SIZE = 0.1


LEN_EXP_BUFFER = 15 if NEXT_OB else 14
OUTPUT_DIM = 4  if NEXT_OB else 20

# /home/dlut/lyn/letter_2/ipp-vae/catnipp-vae/parameters.py
