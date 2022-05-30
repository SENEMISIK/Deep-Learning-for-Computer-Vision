import argparse
from train_helper import *

parser = argparse.ArgumentParser(description='Train and evaluate optical flow model.')
parser.add_argument("-m", "--model-type", default="flownet", type=str)
parser.add_argument("-s", "--size", default=1000, type=int)
parser.add_argument("-p", "--pretrain", default=True, type=bool)

def main():
    args = parser.parse_args()
    pretrain_size = args.size
    model_type = args.model_type
    finetune_size = 520
    pretrain = args.pretrain
    
    train_loader, test_loader, fc_loader = get_data(pretrain_size, finetune_size)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if model_type == "flownet":
        model, pretrain_losses, finetune_losses = train_flownet(fc_loader, train_loader, device, pretrain)
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
        save_results(results, model_type, pretrain_size)
        save_model(model, model_type, pretrain_size)
    
    elif model_type == "raft":
        model, pretrain_losses, finetune_losses = train_raft(fc_loader, train_loader, device, pretrain)
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
        save_results(results, model_type, pretrain_size)
        save_model(model, model_type, pretrain_size)
    else:
        raise(Exception("Invalid model type"))
    

if __name__ == "__main__":
    main()