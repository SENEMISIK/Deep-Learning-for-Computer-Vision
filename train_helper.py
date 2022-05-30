import torch
import torchvision
import numpy as np
from utils import compute_metrics, sequence_loss, InputPadder
from presets import OpticalFlowPresetTrain, OpticalFlowPresetEval
from model import FlowNetS
from multiscaleloss import multiscaleEPE, realEPE
import time
import os

def get_data(pretrain_size, finetune_size):
    fc_transforms = OpticalFlowPresetTrain(crop_size=(368, 496), min_scale=0.1, max_scale=1.0, do_flip=True)
    flying_chairs = torchvision.datasets.FlyingChairs(root=".", split="train", transforms=fc_transforms)
    s_transforms = OpticalFlowPresetTrain(crop_size=(368, 768), min_scale=-0.2, max_scale=0.6, do_flip=True)
    sintel_train = torchvision.datasets.Sintel(root=".", split="train", pass_name="clean", transforms=s_transforms)
    test_transforms = OpticalFlowPresetEval()
    sintel_test = torchvision.datasets.Sintel(root=".", split="train", pass_name="clean", transforms=test_transforms)

    train_ind = np.random.choice(len(sintel_train), finetune_size, replace=False)
    test_ind = np.array(list(set(range(len(sintel_train))) - set(train_ind)))
    fc_ind = np.random.choice(len(flying_chairs), pretrain_size, replace=False)

    sintel_train = torch.utils.data.Subset(sintel_train, train_ind)
    sintel_test = torch.utils.data.Subset(sintel_test, test_ind)
    fc_pretrain = torch.utils.data.Subset(flying_chairs, fc_ind)

    train_loader = torch.utils.data.DataLoader(sintel_train, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(sintel_test, batch_size=10, shuffle=False)
    fc_loader = torch.utils.data.DataLoader(fc_pretrain, batch_size=10, shuffle=True)

    return train_loader, test_loader, fc_loader

def train_flownet_one_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    start = time.time()
    epoch_loss = 0.0
    for i, data_blob in enumerate(train_loader):
        optimizer.zero_grad()
        
        image1, image2, flow_gt, valid_flow_mask = (x.to(device) for x in data_blob)
        image = torch.cat((image1, image2), dim=1)
        
        output = model(image)

        loss = multiscaleEPE(output, flow_gt)
        loss.backward()
        
        optimizer.step()
        
        # Compute epe loss 
        h, w = flow_gt.size()[-2:]
        upsampled_output = torch.nn.functional.interpolate(output[0], (h,w), mode='bilinear', align_corners=False)
        metrics, _ = compute_metrics(upsampled_output, flow_gt)
        epoch_loss += metrics["epe"]
            
    scheduler.step()
    epoch_loss /= len(train_loader)
    print("Epoch", epoch + 1, "finished in", round(time.time() - start, 1), "seconds. Loss:", epoch_loss)
    return epoch_loss

def train_raft_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_train_flow_updates):
    start = time.time()
    epoch_loss = 0.0
    for i, data_blob in enumerate(train_loader):
        optimizer.zero_grad()

        image1, image2, flow_gt, valid_flow_mask = (x.to(device) for x in data_blob)
        flow_predictions = model(image1, image2, num_flow_updates=num_train_flow_updates)

        loss = sequence_loss(flow_predictions, flow_gt, valid_flow_mask)
        metrics, epe_num = compute_metrics(flow_predictions[-1], flow_gt, valid_flow_mask)

        epoch_loss += metrics["epe"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()
        scheduler.step()
    epoch_loss /= len(train_loader)
    print("Epoch", epoch + 1, "finished in", round(time.time() - start, 1), "seconds. Loss:", epoch_loss)
    return epoch_loss

def train_flownet(fc_loader, train_loader, device, pretrain=True):
    pretrain_epochs = 100
    finetune_epochs = 20
    lr = 1e-4
    weight_decay = 4e-4
    model = FlowNetS()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 100, 110], gamma=0.5)

    model.train()
    pretrain_losses = []
    finetune_losses = []
    if pretrain:
        print("Pretraining FlowNet on", len(fc_loader.dataset), "FlyingChairs examples...")
        for epoch in range(pretrain_epochs):
            pretrain_losses.append(train_flownet_one_epoch(model, fc_loader, optimizer, scheduler, device, epoch))
    print("Finetuning Flownet on", len(train_loader.dataset), "Sintel examples...")
    for epoch in range(finetune_epochs):
        finetune_losses.append(train_flownet_one_epoch(model, train_loader, optimizer, scheduler, device, epoch))
    return model, pretrain_losses, finetune_losses

def train_raft(fc_loader, train_loader, device, pretrain=True):
    model = torchvision.models.optical_flow.raft_small()
    model.to(device)
    pretrain_epochs = 20
    finetune_epochs = 10

    lr = 2e-5
    weight_decay = 5e-5
    eps = 1e-8
    num_train_flow_updates = 12
    
    total_steps = finetune_epochs * len(train_loader)
    if pretrain:
        total_steps += pretrain_epochs * len(fc_loader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    
    model.train()
    pretrain_losses = []
    finetune_losses = []
    if pretrain:
        print("Pretraining RAFT on", len(fc_loader.dataset), "FlyingChairs examples...")
        for epoch in range(pretrain_epochs):
            pretrain_losses.append(train_raft_one_epoch(model, fc_loader, optimizer, scheduler, device, epoch, num_train_flow_updates))
    print("Finetuning RAFT on", len(train_loader.dataset), "Sintel examples...")
    for epoch in range(finetune_epochs):
        finetune_losses.append(train_raft_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_train_flow_updates))
    return model, pretrain_losses, finetune_losses
    
def test_flownet(model, test_loader, device):
    model.eval()
    total_epe = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for i, data_blob in enumerate(test_loader):
            image1, image2, flow_gt, valid_flow_mask = (x.to(device) for x in data_blob)
            image = torch.cat((image1, image2), dim=1)

            output = model(image)
            h, w = flow_gt.size()[-2:]
            upsampled_output = torch.nn.functional.interpolate(output, (h,w), mode='bilinear', align_corners=False)
            metrics, _ = compute_metrics(upsampled_output, flow_gt, valid_flow_mask)
            epe = metrics["epe"]
            f1 = metrics["f1"]

            total_epe += epe
            total_f1 += f1

    total_epe /= len(test_loader)
    total_f1 /= len(test_loader)
    
    return total_epe, total_f1

def test_raft(model, test_loader, device, num_test_updates=32):
    model.eval()
    total_epe = 0.0
    total_f1 = 0.0
    

    with torch.no_grad():
        for i, data_blob in enumerate(test_loader):
            image1, image2, flow_gt = (x.to(device) for x in data_blob)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_predictions = model(image1, image2, num_flow_updates=num_test_updates)
            flow_pred = flow_predictions[-1]
            flow_pred = padder.unpad(flow_pred)
            
            metrics, _ = compute_metrics(flow_pred, flow_gt)
            epe = metrics["epe"]
            f1 = metrics["f1"]

            total_epe += epe
            total_f1 += f1

    total_epe /= len(test_loader)
    total_f1 /= len(test_loader)
    
    return total_epe, total_f1

def save_results(results, model_type, pretrain_size):
    path = os.path.join("results", model_type + str(pretrain_size) + "_results.txt")
    torch.save(results, path)
    
def save_model(model, model_type, pretrain_size):
    path = os.path.join("results", model_type + str(pretrain_size) + "_params.pt")
    torch.save(model.state_dict(), path)