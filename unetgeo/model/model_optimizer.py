import torch
def get_optimizer(tparam, model):
    """
        The geometric-based approach is not a learned model.
        So, here torch.nn.Linear(10, 20) we only need to give a 
        simple linear layer to run the model. 
    """
    map_optimizers_name_to_type = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
    }
    optimizer_type = map_optimizers_name_to_type["adam"]

    if model.name == "RelationTagging":
        fake_model1 = torch.nn.Linear(10, 20)
        param_list = [
            {
                "params": filter(
                    lambda p: p.requires_grad, fake_model1.parameters()
                ),
                "lr": 0.00004,
            }
        ]
        optimizer = optimizer_type(param_list, lr=0.00004, weight_decay=0)

    else:
        raise NotImplementedError

    return optimizer