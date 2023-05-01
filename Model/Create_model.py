from .Unet import Unet

def Create_model(args):
    if (args.Network == "Unet"):
        net =  Unet(len(args.Input), len(args.Output))
        return net