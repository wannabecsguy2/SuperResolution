import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.utils import save_image

plt.style.use('ggplot')

def psnr(label, outputs, max_val = 1.):
    '''
    Compute Peak Signal to Noise Ratio [Higher the better]
    PNSR = 20 * log_10(MAXp) - 20 * log_10(MSE)
    '''
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean(diff)**2)
    if rmse == 0:
        return 100
    else:
        PSNR = 20* math.log10(max_val/rmse)
        return PSNR

def save_plot(train_loss, val_loss, train_psnr, val_psnr):
    plt.figure(figsize = (10, 7))
    plt.plot(train_loss, color = 'orange', label = 'Train Loss')
    plt.plot(val_loss, color = 'red', label = 'Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.png')
    plt.close()

    #PNSR plots
    plt.figure(figsize = (10, 7))
    plt.plot(train_psnr, color = 'orange', label = 'Train PSNR')
    plt.plot(val_psnr, color = 'red', label = 'Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig('../outputs/psnr.png')
    plt.close()

def save_model_state(model):
    print('Saving model...\n')
    torch.save_model(model.state_dict(), '../outputs/model.pth')
    print('Model Saved')

def save_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs+1,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : criterion 
    }, f"../outputs/model_ckpt.pth")

def save_validation_results(outputs, epoch, batch_iter):
    '''
    Function to save the validation reconstructed images
    '''
    save_image(outputs, f'../outputs/valid_results/val_sr_{epoch}_{batch_iter}.png')
