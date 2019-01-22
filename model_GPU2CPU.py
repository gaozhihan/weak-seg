import torch

model_dict = torch.load("st_01/models/st_01_top_val_rec_SEC_31_31.pth")


for item in model_dict.keys():
    model_dict[item] = model_dict[item].cpu()

torch.save(model_dict, "st_01/models/st_01_top_val_rec_SEC_31_31_cpu.pth")
