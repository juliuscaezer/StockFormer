# src/train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os

# Import all the components you've built
import config
from dataset import StockDataset
from components.model import Stockformer
from utils.loss_functions import stock_tanh_loss

def run_training():
    """Main function to run the model training and validation."""
    
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create a directory to save models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/best_stockformer_model.pth"

    # --- 2. Data Loading ---
    full_dataset = StockDataset(
        csv_file_path=config.PROCESSED_DATA_PATH,
        seq_len=config.SEQ_LEN,
        target_ticker=config.TARGET_TICKER
    )
    
    # Split dataset into training and validation sets (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Data loaded. Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # --- 3. Model, Optimizer, and Scheduler ---
    model = Stockformer(
        num_stocks=config.NUM_STOCKS,
        seq_len=config.SEQ_LEN,
        embed_size=config.EMBED_SIZE,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT # Added missing hyperparameters
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    print("\nStarting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        
        for x_batch, t_batch, y_batch in train_loader:
            x_batch, t_batch, y_batch = x_batch.to(device), t_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(x_batch, t_batch)
            loss = stock_tanh_loss(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, t_batch, y_batch in val_loader:
                x_batch, t_batch, y_batch = x_batch.to(device), t_batch.to(device), y_batch.to(device)
                predictions = model(x_batch, t_batch)
                loss = stock_tanh_loss(predictions, y_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Update the learning rate (custom) scheduler
        scheduler.step(avg_val_loss)
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with validation loss: {best_val_loss:.6f}")

    print("\nTraining finished!")
    print(f"Best model saved at {best_model_path}")

if __name__ == '__main__':
    run_training()