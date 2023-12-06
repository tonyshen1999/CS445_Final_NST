import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer import TransformerNet
from vgg import Vgg16

# GLOBAL SETTINGS
# parameters borrowed from https://github.com/rrmina/fast-neural-style-pytorch 
# for initial testing, slightly increase batch size for faster training
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "dataset"
NUM_EPOCHS = 1
STYLE_IMAGE_PATH = "images/mosaic.jpg"
BATCH_SIZE = 8
CONTENT_WEIGHT = 17 # 17
STYLE_WEIGHT = 50 # 25
ADAM_LR = 0.001
SAVE_MODEL_PATH = "models/"
SAVE_IMAGE_PATH = "images/out/"
SAVE_MODEL_EVERY = 500 # 4,000 Images with batch size 8
SEED = 35
PLOT_LOSS = 1


def train():
    # preparation before training
    device = torch.device("cuda")
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda z: z.mul(255))
    ])
    
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    transformer_net = TransformerNet().to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    adam = Adam(transformer_net.parameters(), lr = ADAM_LR)
    mse_loss = torch.nn.MSELoss()
    
    # As gram matrix is a matrix with fixed dimensions c_j x c_j, 
    # no need for resizing
    style_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda z: z.mul(255))
    ])
    style = style_trans(utils.load_image(STYLE_IMAGE_PATH))
    style = style.repeat(BATCH_SIZE, 1, 1, 1).to(device)
    
    style_features = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in style_features]
    
    # start training with adam
    for e in range(NUM_EPOCHS):
        transformer_net.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            adam.zero_grad()
            
            x = x.to(device)
            y = transformer_net(x)
            x = utils.normalize_batch(x)
            y = utils.normalize_batch(y)
            x_features = vgg(x)
            y_features = vgg(y)
            
            # According to paper using relu2_2 for content loss
            # has less distortions than higher level features
            content_loss = mse_loss(x_features.relu2_2, y_features.relu2_2) * CONTENT_WEIGHT
            style_loss = 0.
            for ft_y, gm_s in zip(y_features, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_s[:n_batch,:,:], gm_y)
            style_loss *= STYLE_WEIGHT
            
            tot_loss = content_loss + style_loss
            tot_loss.backward()
            adam.step()
            
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            
            if (batch_id + 1) % SAVE_MODEL_EVERY == 0 :
                msg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset), 
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(msg)
                transformer_net.eval().cpu()
                ckpt_model_filename =  "ckpt_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(SAVE_MODEL_PATH, ckpt_model_filename)
                torch.save(transformer_net.state_dict(), ckpt_model_path)
                transformer_net.to(device).train()
            
    # save the model
    transformer_net.eval().cpu()
    model_filename =  "epoch_" + str(NUM_EPOCHS) + "_" + str(
        time.ctime()).replace(' ', '_').replace(':', '-') + "_" + str(
            CONTENT_WEIGHT) + "_" + str(STYLE_WEIGHT) + ".model"
    model_path = os.path.join(SAVE_MODEL_PATH, model_filename)
    torch.save(transformer_net.state_dict(), model_path)
    print("\nDone, trained model saved at", model_path)

train()