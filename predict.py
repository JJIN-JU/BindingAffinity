import torch

from models.bilinear_model import BilinearAffinityZ_MLP_LR
from models.attnbind_model import AttnBind

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):

    model = AttnBind(lig_dim=11, cav_dim=37)

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model
