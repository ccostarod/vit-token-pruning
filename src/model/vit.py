import timm


def create_vit_model(
    num_classes: int = 102,
    pretrained: bool = True,
    model_name: str = "vit_base_patch16_224.augreg_in21k",
):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model
