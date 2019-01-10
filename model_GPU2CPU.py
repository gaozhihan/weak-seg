import torch

model_dict = torch.load("multi_scale/models/st_top_val_acc_my_resnet_9_9.pth")


for item in model_dict.keys():
    model_dict[item] = model_dict[item].cpu()

torch.save(model_dict, "multi_scale/models/st_top_val_acc_my_resnet_09_01_cpu.pth")
