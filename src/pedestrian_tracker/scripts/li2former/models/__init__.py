from .drow_net.drow_net import DrowNet
from .li2former import Li2Former
from .dr_spaam import DrSpaam

__MODEL__ = {"DROW": DrowNet, "Li2Former": Li2Former, "DR-SPAAM": DrSpaam}


def buildModel(config):
    loss_cfg = config("PIPELINE")["LOSS"]["KWARGS"]
    model_cfg = config("MODEL")
    return __MODEL__[model_cfg["TYPE"]](
        loss_kwargs=loss_cfg, model_kwargs=model_cfg["KWARGS"]
    )
