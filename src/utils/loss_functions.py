import torch

def stock_tanh_loss(y_pred, y_true):
    """
    Custom loss function that calculates the negative ROI from a trading strategy.
    The strategy invests a percentage of the portfolio based on tanh(prediction).
    - y_pred: The model's raw output. Shape: [batch_size, 1]
    - y_true: The actual LogPercentChange. Shape: [batch_size, 1]
    """
    # The investment amount is determined by the hyperbolic tangent of the prediction
    investment_pct = torch.tanh(y_pred)
    
    # The log return of the strategy is (investment_pct * y_true)
    # We sum the log returns over the batch
    total_log_return = torch.sum(investment_pct * y_true)
    
    # The goal of training is to MAXIMIZE the return.
    # Since optimizers work by MINIMIZING a loss, we return the NEGATIVE of the profit.
    return -total_log_return