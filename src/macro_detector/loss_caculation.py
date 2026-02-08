import torch

def Loss_Calculation(outputs, batch):
    return MAE_Loss(outputs, batch)

def MSE_Loss(outputs, batch):
    return torch.mean((outputs - batch)**2, dim=(1, 2)) 

def MAE_Loss(outputs, batch):
    return torch.mean(torch.abs(outputs - batch), dim=(1, 2))
