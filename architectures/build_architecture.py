"""
To select the architecture based on a config file we need to ensure
we import each of the architectures into this file. Once we have that
we can use a keyword from the config file to build the model.
"""
######################################################################
def build_architecture(config):
    if config["model_name"] == "segformer3d":
        from .segformer3d import build_segformer3d_model

        model = build_segformer3d_model(config)

        return model
    
    if config["model_name"] == "trans3dUNET":
        from .transUNET3D import build_transunet3d_model
        model = build_transunet3d_model()
        return model
    
    if config["model_name"] == "swin3d":
        from .swin3DTranformer import build_swin3d_segmentation_model
        model = build_swin3d_segmentation_model()
        return model

    if config["model_name"] == "nnformer":
        from .swin3DTranformer import build_swin3d_segmentation_model
        model = build_swin3d_segmentation_model()
        return model
    else:
        return ValueError(
            "specified model not supported, edit build_architecture.py file"
        )
