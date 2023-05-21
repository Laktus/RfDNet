from pcdet.models.detectors.cagroup3d import CAGroup3D

__all__ = {
    'CAGroup3D': CAGroup3D,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model