from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
def combine_images(columns, space, images, save_path):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([Image.open(image).width for image in images])
    height_max = max([Image.open(image).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(save_path)
# def combine_images_ffmpeg(columns, space, images, save_path):
#     rows = len(images) // columns
#     if len(images) % columns:
#         rows += 1
#     width_max = max([Image.open(image).width for image in images])
#     height_max = max([Image.open(image).height for image in images])
#     background_width = width_max*columns + (space*columns)-space
#     background_height = height_max*rows + (space*rows)-space
#     ## this is used for ffmpeg. Even numbers are needed.
#     if background_width % 2 == 1:
#         background_width += 1
#     if background_height % 2 == 1:
#         background_height += 1
#     background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
#     x = 0
#     y = 0
#     for i, image in enumerate(images):
#         img = Image.open(image)
#         x_offset = int((width_max-img.width)/2)
#         y_offset = int((height_max-img.height)/2)
#         background.paste(img, (x+x_offset, y+y_offset))
#         x += width_max + space
#         if (i+1) % columns == 0:
#             y += height_max + space
#             x = 0
#     background.save(save_path)

class SuperMAGDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = torch.load(X_path)
        self.Y = torch.load(Y_path)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], "Y": self.Y[idx]}
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0,
                 path="C:/Users/feixi/Desktop/deltaB/model/model_state.pth",
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.2f} --> {val_loss:.2f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def compute_accuracy(model, data_loader, scalerY, Image_folder, i, j, plot=True):
    with torch.no_grad():
        pred_means, pred_var, lls = model.predict(data_loader)
        
        test_Y_OG = scalerY.inverse_transform(data_loader.dataset.Y.numpy().reshape(-1,1))
        pred_Y_OG = scalerY.inverse_transform(pred_means.mean(0).cpu().numpy().reshape(-1,1))
        if plot:
            plt.plot(test_Y_OG, color = "black")
            plt.plot(pred_Y_OG, color="red")
            plt.savefig(Image_folder+ 'Epoch_%03d_%03d.png' % (i,j), facecolor = 'white')
            plt.close()
        rmse = np.sqrt(np.nanmean((test_Y_OG - pred_Y_OG)**2))
        MAE = mean_absolute_error(test_Y_OG, pred_Y_OG)
        return rmse, MAE, lls
    
class SuperMAG_Num(Dataset):
    def __init__(self, Y_path):
        self.Y = torch.load(Y_path)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {"Y": self.Y[idx]}
class SuperMAGDataset_fromtensor(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], "Y": self.Y[idx]}

class SuperMAGDataset_test(Dataset):
    def __init__(self, storm_X_tensor, storm_Y_tensor):
        self.X = storm_X_tensor
        self.Y = storm_Y_tensor
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], "Y": self.Y[idx]}

def sigma_GMM(pred_mean, pred_var):
    #\sigma = [\sigma1, \sigma2, ...]
    sigma = np.sqrt(pred_var.cpu().numpy()).reshape(-1,1).reshape(pred_var.shape)
    # \bar{\sigma^2}
    bar_sigma_sq = (sigma**2).mean(0)
    ## \mu = [\mu1. \mu2, ...]
    mu = pred_mean.cpu().numpy().reshape(-1,1).reshape(pred_mean.shape)
    ## \bar(\mu^2)
    bar_mu_sq = (mu**2).mean(0)
    ## (\bar{\mu})^2
    sq_bar_mu = (mu.mean(0))**2
    ## the standard divation of a GMM f = \sum 1\{n} f_{i} with f_{i} being Gaussian.
    sigma_f = np.sqrt(bar_sigma_sq + bar_mu_sq - sq_bar_mu)
    return sigma_f

def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def compute_hss(true_values, predictions, threshold):
    """
    Compute Heidke Skill Score based on a given threshold.
    
    Parameters:
    - true_values: List or array of true values.
    - predictions: List or array of predicted values.
    - threshold: Threshold value to determine event occurrence.
    
    Returns:
    - HSS: Heidke Skill Score.
    """

    # Convert numerical values to binary based on threshold
    true_binary = pd.DataFrame((true_values >= threshold).astype(int))
    pred_binary = pd.DataFrame((predictions >= threshold).astype(int))
    pred_binary.index = true_binary.index

    # Calculate confusion matrix values
    a = np.sum((true_binary == 1) & (pred_binary == 1))
    b = np.sum((true_binary == 0) & (pred_binary == 1))
    c = np.sum((true_binary == 1) & (pred_binary == 0))
    d = np.sum((true_binary == 0) & (pred_binary == 0))

    # Compute Heidke Skill Score
    denominator = (a+b)*(b+d) + (a+c)*(c+d)

    return round(2*(a*d - b*c) / denominator,2)

def compute_hss_count(true_values, predictions, threshold):
    """
    Compute Heidke Skill Score based on a given threshold.
    
    Parameters:
    - true_values: List or array of true values.
    - predictions: List or array of predicted values.
    - threshold: Threshold value to determine event occurrence.
    
    Returns:
    - HSS: Heidke Skill Score.
    """

    # Convert numerical values to binary based on threshold
    true_binary = pd.DataFrame((true_values >= threshold).astype(int))
    pred_binary = pd.DataFrame((predictions >= threshold).astype(int))
    pred_binary.index = true_binary.index

    # Calculate confusion matrix values
    a = np.sum((true_binary == 1) & (pred_binary == 1))
    b = np.sum((true_binary == 0) & (pred_binary == 1))
    c = np.sum((true_binary == 1) & (pred_binary == 0))
    d = np.sum((true_binary == 0) & (pred_binary == 0))

    # Compute Heidke Skill Score
    denominator = (a+b)*(b+d) + (a+c)*(c+d)
    return (a,b,c,d)
#     return round(2*(a*d - b*c) / denominator,2)