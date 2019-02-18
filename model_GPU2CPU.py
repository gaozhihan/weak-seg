import torch

model_dict = torch.load("multi_scale/models/st_pure_gray_top_val_acc_my_resnet_15_15.pth")


for item in model_dict.keys():
    model_dict[item] = model_dict[item].cpu()

torch.save(model_dict, "multi_scale/models/st_pure_gray_top_val_acc_my_resnet_15_15_cpu.pth")
