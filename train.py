import argparse
from train_helper import *

parser = argparse.ArgumentParser(description='Train and evaluate optical flow model.')
parser.add_argument("-m", "--model-type", default="flownet", type=str)
parser.add_argument("-fc", "--fc-size", default=1000, type=int)
parser.add_argument("-s", "--sintel-size", default=200, type=int)
parser.add_argument("-p", "--pretrain", default=1, type=int)
parser.add_argument("-a", "--augment", default=1, type=int)

def main():
    args = parser.parse_args()
    pretrain_size = args.fc_size
    finetune_size = args.sintel_size
    model_type = args.model_type
    pretrain = bool(args.pretrain)
    augment = bool(args.augment)
    
    train_loader, test_loader, fc_loader = get_data(pretrain_size, finetune_size, augment)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if model_type == "flownet":
        model, pretrain_losses, finetune_losses = train_flownet(fc_loader, train_loader, device, augment, pretrain)
        epe, f1 = test_flownet(model, test_loader, device)
        print("Test:")
        print("Epe: ", epe)
        print("F1: ", f1)
        results = {
            "pretrain_losses": pretrain_losses, 
            "finetune_losses": finetune_losses,
            "epe": epe,
            "f1": f1
        }
        if not pretrain:
            pretrain_size = 0
        save_results(results, model_type, pretrain_size, augment)
        save_model(model, model_type, pretrain_size, augment)
    
    elif model_type == "raft":
        model, pretrain_losses, finetune_losses = train_raft(fc_loader, train_loader, device, augment, pretrain)
        epe, f1 = test_raft(model, test_loader, device)
        print("Test:")
        print("Epe: ", epe)
        print("F1: ", f1)
        results = {
            "pretrain_losses": pretrain_losses, 
            "finetune_losses": finetune_losses,
            "epe": epe,
            "f1": f1
        }
        if not pretrain:
            pretrain_size = 0
        save_results(results, model_type, pretrain_size, augment)
        save_model(model, model_type, pretrain_size, augment)
    else:
        raise(Exception("Invalid model type"))
    

if __name__ == "__main__":
    main()