# StockFormer

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A transformer-based deep learning model for multivariate stock price prediction, specifically designed to predict WTI crude oil prices using temporal patterns from related energy sector stocks.

## 🚀 Features

- **Hybrid Architecture**: Combines 1D-CNN feature extraction with transformer attention mechanisms
- **Time2Vec Encoding**: Advanced temporal embeddings that capture both linear and periodic time patterns
- **Trading-Focused Loss**: Custom loss function optimized for actual trading returns (ROI) rather than traditional MSE
- **Multi-Stock Context**: Leverages 7 related oil/energy stocks for richer predictive context
- **GPU Acceleration**: Supports CUDA for faster training and inference

## 📊 Model Architecture

StockFormer employs a sophisticated multi-component architecture:

1. **1D-CNN Layer**: Extracts local patterns from multivariate stock data
2. **Time2Vec Encoding**: Learns temporal representations with both linear and periodic components
3. **Transformer Encoder**: Captures long-range dependencies with self-attention (3 layers, 8 heads)
4. **Output Layer**: Maps encoded features to price movement predictions

### Input Data
- **Stocks**: BP, COP, CVX, EOG, PBR, WTI, XOM (7 energy sector stocks)
- **Sequence Length**: 48 hours
- **Target**: WTI crude oil price movements

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for complete dependency list.

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/StockFormer.git
   cd StockFormer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv stockformer-env
   source stockformer-env/bin/activate  # On Windows: stockformer-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **GPU Support** (optional)
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🚀 Quick Start

### Training the Model

```bash
python src/train.py
```

The training script will:
- Load processed data from `data/processed/oil_stocks_hourly_processed.csv`
- Train for 50 epochs with early stopping
- Save the best model to `models/best_stockformer_model.pth`

### Using a Trained Model

```python
import torch
from src.components.model import Stockformer
from src import config

# Initialize model
model = Stockformer(
    num_stocks=config.NUM_STOCKS,
    seq_len=config.SEQ_LEN,
    embed_size=config.EMBED_SIZE,
    num_encoder_layers=config.NUM_ENCODER_LAYERS,
    num_heads=config.NUM_HEADS,
    dropout=config.DROPOUT
)

# Load trained weights
model.load_state_dict(torch.load('models/best_stockformer_model.pth'))
model.eval()

# Make predictions
# predictions = model(x_batch, t_batch)
```

### Data Exploration

Explore the dataset and model performance:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## 📁 Project Structure

```
StockFormer/
├── data/
│   ├── raw/                    # Original stock data
│   └── processed/              # Preprocessed data files
├── models/                     # Saved model checkpoints
├── notebooks/                  # Jupyter notebooks
│   └── data_exploration.ipynb  # Data analysis and visualization
├── src/
│   ├── components/             # Model components
│   │   ├── __init__.py
│   │   ├── attention.py        # Custom attention mechanisms
│   │   ├── model.py           # Main Stockformer model
│   │   └── time2vec.py        # Time2Vec temporal encoding
│   ├── utils/
│   │   ├── __init__.py
│   │   └── loss_functions.py  # Custom loss functions
│   ├── __init__.py
│   ├── config.py              # Model hyperparameters
│   ├── dataset.py             # Data loading and preprocessing
│   └── train.py              # Training script
├── .gitignore
├── requirements.txt
├── README.md
└── WARP.md                    # WARP AI development guide
```

## ⚙️ Configuration

Model hyperparameters are centralized in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEQ_LEN` | 48 | Input sequence length (hours) |
| `EMBED_SIZE` | 512 | Model embedding dimension |
| `NUM_ENCODER_LAYERS` | 3 | Transformer encoder layers |
| `NUM_HEADS` | 8 | Multi-head attention heads |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-6 | Initial learning rate |
| `EPOCHS` | 50 | Maximum training epochs |

## 📈 Training Details

### Loss Function
StockFormer uses a custom `stock_tanh_loss` that optimizes trading returns:
- Investment percentage determined by `tanh(prediction)`
- Maximizes actual ROI rather than minimizing prediction error
- Directly applicable to trading strategies

### Data Splitting
- **Training**: 80% (temporal split)
- **Validation**: 20%
- **Strategy**: Time-series aware splitting to prevent data leakage

### Optimization
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping**: Based on validation loss

## 🔬 Model Components

### Time2Vec Encoding
Learnable temporal representations that combine:
- **Linear term**: `w₀ * t + φ₀`
- **Periodic terms**: `sin(wᵢ * t + φᵢ)`

### Custom Attention
Currently implements standard PyTorch MultiheadAttention. For full replication, replace with ProbSparse attention mechanism.

### Dataset Class
Handles sliding window sequences from processed stock data with proper temporal alignment.

## 🙏 Acknowledgments

- Time2Vec implementation based on [Kazemi et al., 2019](https://arxiv.org/abs/1907.05321)
- Inspired by transformer architectures for time series forecasting
- Built with PyTorch framework