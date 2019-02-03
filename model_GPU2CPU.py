import torch

model_dict = torch.load("st_resnet/models/res_from_mul_scale_resnet_cue_01_hard_snapped_my_resnet.pth")


for item in model_dict.keys():
    model_dict[item] = model_dict[item].cpu()

torch.save(model_dict, "st_resnet/models/res_from_mul_scale_resnet_cue_01_hard_snapped_my_resnet_cpu.pth")
