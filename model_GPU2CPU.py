import torch

model_dict = torch.load("st_resnet/models/res_wsc_ft_gray_color_0221_0222_my_resnet.pth")


for item in model_dict.keys():
    model_dict[item] = model_dict[item].cpu()

torch.save(model_dict, "st_resnet/models/res_wsc_ft_gray_color_0221_0222_my_resnet_cpu.pth")
