import torch
import tqdm
import gpytorch
from gpytorch.mlls import DeepApproximateMLL
import pickle
from torch.utils.data import DataLoader
import re
import os
import sys
from gpytorch.mlls import VariationalELBO
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.utils import *
from src.model import *

path = root_path+"/data/train/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
QoIs = "dBH"
yeojohnson_lmbda = 1
window_max = 10
res_folder = path + 'Maximum_storms/%sT/' % window_max
with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
    scalerX = pickle.load(f)
with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
    scalerY = pickle.load(f)
val_dataset = SuperMAGDataset(res_folder + 'val_X_%s.pt' % QoIs, res_folder + 'val_Y_%s.pt' % QoIs)
val_loader = DataLoader(val_dataset,
                        batch_size=1024)
X_shape = val_dataset.X.shape[1]

model = DeepGaussianProcess(X_shape,
                            DEVICE,
                            num_hidden1_dims = 20,
                            num_hidden2_dims = 10,
                            num_hidden3_dims = 10)

if torch.cuda.is_available():
    model = model.cuda()
train_X_file = sorted([x for x in os.listdir(res_folder) if bool(re.match(r'^train_X_%s' % QoIs, x))])
train_Y_file = sorted([x for x in os.listdir(res_folder + "%s/" % QoIs) if bool(re.match(r'^train_Y', x))])
train_num = 0
for j in range(len(train_X_file)):
    train_dataset = SuperMAG_Num(res_folder + "%s/" % QoIs + train_Y_file[j])
    train_num += train_dataset.__len__()
# this is for running the notebook in our testing framework
num_epochs = 50
num_samples = 15
tracking_folder = root_path+"/data/train/training_tracking/Maximum/%s/%sT/" % (QoIs, window_max)
model_earlystopping = root_path+"/data/train/model/model_state_%s_%sT.pth" % (QoIs, window_max)
model_archive = root_path+"/data/train/model/Maximum/%s/" % QoIs
os.makedirs(tracking_folder, exist_ok = True)
os.makedirs(model_archive, exist_ok = True)
early_stopping = EarlyStopping(patience=7,verbose=False,
                              path=model_earlystopping)
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
train_pt_file = []
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_num))
j = 0
train_dataset = SuperMAGDataset(res_folder + train_X_file[j],
                                res_folder + "%s/" % QoIs + train_Y_file[j])
train_loader = DataLoader(train_dataset,
                          batch_size=1024,
                          shuffle=True)
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
    for batch in minibatch_iter:
        x_batch = batch['X'].to(DEVICE)
        y_batch = batch['Y'].to(DEVICE)
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
    validation_rmse, MAE, lls = compute_accuracy(model, val_loader, scalerY,
                                      tracking_folder, i+1,j+1,
                                      plot=True)
    torch.save(model.state_dict(), model_archive + "model_state_%s_%03d_%03d.pth" % (QoIs, i+1, j+1))
    print("Epoch %s/%s;" % (i+1, num_epochs)
          + " Group: %s/%s;" % (j+1, len(train_X_file))
          + " RMSE: %.1f;" % validation_rmse
          + ' MAE: %.1f;' % MAE
          + " learning rate: %.5f" % optimizer.param_groups[0]["lr"])
    scheduler.step(-lls)
    early_stopping(-lls, model)
    if early_stopping.early_stop:
        print('Early stopping')
        break

