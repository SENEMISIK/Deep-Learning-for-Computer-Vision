import torch
import torchvision
import os
from utils import compute_metrics
import argparse

parser = argparse.ArgumentParser(description='Test optical flow model on Kitti')
parser.add_argument("-t", "--model-type", default="flownet", type=str)
parser.add_argument("-m", "--model", default=".", type=str)

class TKitti(torchvision.datasets.KittiFlow):
    def __init__(self, root):
        super().__init__(root=root)
        
    def __getitem__(self, index):
        img1, img2, flow, valid_flow_mask = super().__getitem__(index)
        img1 = torchvision.transforms.ToTensor()(img1)
        img2 = torchvision.transforms.ToTensor()(img2)
        return img1, img2, flow, valid_flow_mask

def test():
    kitti = TKitti("./Kitti")
    test_loader = torch.utils.data.DataLoader(kitti, batch_size=1, shuffle=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    total_epe = 0.0
    total_f1 = 0.0
    total_1px = 0.0
    total_3px = 0.0
    total_5px = 0.0
    num = 0
    if args.model_type == "flownet":
        model = FlowNetS()    
        model.load_state_dict(torch.load(args.model))
        model.eval()   
        model.to(device)   
        with torch.no_grad():
            for i, (img1, img2, target) in tqdm(enumerate(val_loader)):
                image = torch.cat((img1, img2), dim=1).to(device)
                label = target.to(device)
                output = model(image)
                img_size = kitti[i][0].shape[1:]
                output = torch.nn.functional.interpolate(output, size=img_size, mode="bilinear", align_corners=False).squeeze()
                metrics, _ = compute_metrics(output, label)
                epe = metrics["epe"]
                f1 = metrics["f1"]
                total_1px += metrics['1px']
                total_3px += metrics['3px']
                total_5px += metrics['5px']
                total_epe += epe
                total_f1 += f1
                num += 1
    elif args.model_type == "raft":
        model = torchvision.models.optical_flow.raft_small()
        model.load_state_dict(torch.load(args.model))
        model.eval()   
        model.to(device)
        num_test_updates = 32
        with torch.no_grad():                                         
            for i, (image1, image2, flow_gt) in tqdm(enumerate(val_loader)):
                image1.to(device)
                image2.to(device)
                flow_gt.to(device)
                
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)              
                      
                flow_predictions = model(image1, image2, num_flow_updates=num_test_updates)
                flow_pred = flow_predictions[-1]
                flow_pred = padder.unpad(flow_pred)
                
                metrics, _ = compute_metrics(flow_pred, flow_gt)
                epe = metrics["epe"]
                f1 = metrics["f1"]
                total_1px += metrics['1px']
                total_3px += metrics['3px']
                total_5px += metrics['5px']
                total_epe += epe
                total_f1 += f1
                num += 1
    else:
        raise(Exception("Invalid model type"))                                 
    
                                         total_epe /= num
    total_f1 /= num
    total_1px /= num
    total_3px /= num
    total_5px /= num
    return total_epe, total_f1, total_1px, total_3px, total_5px 

if __name__ == "__main__":
    epe, f1, px1, px3, px5 = test()
    print("Test:")
    print("Epe: ", epe)
    print("F1: ", f1)
    print("1px: ", px1)
    print("3px: ", px3)
    print("5px: ", px5)
    results = {
        "pretrain_losses": pretrain_losses, 
        "finetune_losses": finetune_losses,
        "epe": epe,
        "f1": f1,
        "1px:": px1,
        "3px:": px3,
        "5px:": px5
    }
    path = os.path.join("results", args.model + "_kitti_test.txt")
    torch.save(results, path)
    