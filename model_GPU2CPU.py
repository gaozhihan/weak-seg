import torch

model_dict = torch.load("models/0506/top_val_rec_SEC_06.pth")


for item in model_dict.keys():
    model_dict[item] = model_dict[item].cpu()

torch.save(model_dict, "models/0506/top_val_rec_SEC_06_CPU.pth")
