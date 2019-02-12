import torch

model_dict = torch.load("multi_scale/models/st_rand_gray_top_val_acc_my_resnet_11_11.pth")


for item in model_dict.keys():
    model_dict[item] = model_dict[item].cpu()

torch.save(model_dict, "multi_scale/models/st_rand_gray_top_val_acc_my_resnet_11_11_cpu.pth")
