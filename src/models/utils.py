import torch

def load_checkpoint(checkpoint, model: torch.nn.Module) -> None:
    """
        args:
            checkpoint - e.g. torch.load(...) result.
            model: torch.nn.Module - model's architecture for
                checkpoint to be loaded in.
        returns:
            None - passed model will have loaded weights.
    """

    model.load_state_dict(checkpoint['state_dict'])

def save_checkpoint(state: dict, path: str) -> None:
    """
        args:
            state: dict - dictionary of { state_dict: model.state_dict(),
                optimizer: optimizer.state_dict() }
            path: str - path for model to be saved to
    """
    
    torch.save(state, path)