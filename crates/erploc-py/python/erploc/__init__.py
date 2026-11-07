"""Self-localisation from a 360 degree (equirectangular) camera and planar markers, and marker placement."""

from ._erploc import (  # noqa: F401
    Bundle,
    BundleState,
    Camera,
    NumericalError,
    Observation,
    Placement,
    Pose,
    Resection,
    Scene,
    __version__,
    bundle_adjust,
    evaluate,
    fisher_camera,
    init_bundle,
    observations_from_eqar,
    place_markers,
    predict_corners,
    resection,
    sigma_pos,
    upright_marker,
    wall_candidates,
)

__all__ = [
    "Bundle", "BundleState", "Camera", "NumericalError", "Observation", "Placement", "Pose", "Resection", "Scene",
    "__version__", "bundle_adjust", "evaluate", "fisher_camera", "init_bundle", "observations_from_eqar",
    "place_markers", "predict_corners", "resection", "sigma_pos", "upright_marker", "wall_candidates",
]
