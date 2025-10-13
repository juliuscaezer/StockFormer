# src/config.py

# -- Data Configuration --
PROCESSED_DATA_PATH = "./data/processed/oil_stocks_hourly_processed.csv"
TARGET_TICKER = 'WTI' # The stock we want to predict

# -- Model Hyperparameters --
SEQ_LEN = 48          # Number of time steps in each input sequence (e.g., 48 hours)
EMBED_SIZE = 512      # Embedding size / model dimension
NUM_ENCODER_LAYERS = 3 # Number of encoder layers
NUM_HEADS = 8         # Number of attention heads (must be a divisor of EMBED_SIZE)
NUM_STOCKS = 7        # Number of stocks in your multivariate input (BP, COP, CVX, EOG, PBR, WTI, XOM)
DROPOUT = 0.1

# -- Training Hyperparameters --
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-6 # Transformers are sensitive to learning rate