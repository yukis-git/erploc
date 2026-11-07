"""Type stubs for erploc.

Invalid values raise ``ValueError`` (``TypeError`` for the wrong kind of object). An integer argument that is negative where the
parameter is unsigned (``width``, ``n_max``, ``k_min``, ``max_iter``, ...) or does not fit the 64-bit type raises ``OverflowError``.
"""

from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str

NoiseModel = Literal["tangent", "pixel"]
Loss = Literal["linear", "huber"]
Dof = Literal["yaw", "full"]
Method = Literal["greedy", "ift"]

#: ``{id: (position (3,), rotation (3, 3))}``; the rotation columns are the marker x (right), y (up) and normal axes.
Markers = Mapping[int, Tuple[ArrayLike, ArrayLike]]
#: ``Observation``, an ``eqar.Marker`` or its ``to_dict()`` (anything with ``id`` and ``corners``).
ObservationLike = Any

class NumericalError(RuntimeError):
    """A computation failed numerically (singular system, no convergence, no seed frame, ...)."""

class Camera:
    """Equirectangular camera: ``width == 2 * height``; ``lam_sign`` is +1 or -1 (direction of increasing column)."""

    def __init__(self, width: int, height: Optional[int] = None, lam_sign: int = 1) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def lam_sign(self) -> int: ...

class Pose:
    """Camera pose; ``rotation`` is camera-to-world, ``R = Rz(yaw) Ry(pitch) Rx(roll)`` unless given."""

    def __init__(
        self,
        position: ArrayLike,
        yaw: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        rotation: Optional[ArrayLike] = None,
    ) -> None: ...
    @property
    def position(self) -> NDArray[np.float64]: ...
    @property
    def yaw(self) -> float: ...
    @property
    def roll(self) -> float: ...
    @property
    def pitch(self) -> float: ...
    @property
    def rotation(self) -> NDArray[np.float64]: ...

class Observation:
    """One detected marker: ``corners`` (4, 2) ERP pixels (TL, TR, BR, BL), per-corner validity and sigma scale."""

    def __init__(
        self,
        id: int,
        corners: ArrayLike,
        corner_valid: Optional[ArrayLike] = None,
        corner_sigma_scale: Optional[ArrayLike] = None,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def corners(self) -> NDArray[np.float64]: ...
    @property
    def corner_valid(self) -> List[bool]: ...
    @property
    def corner_sigma_scale(self) -> NDArray[np.float64]: ...

class Resection:
    @property
    def pose(self) -> Pose: ...
    @property
    def position(self) -> NDArray[np.float64]: ...
    @property
    def yaw(self) -> float: ...
    @property
    def cov(self) -> NDArray[np.float64]: ...
    @property
    def info(self) -> NDArray[np.float64]: ...
    @property
    def sigma_pos(self) -> float: ...
    @property
    def rms_px(self) -> float: ...
    @property
    def n_markers(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def dof(self) -> Dof: ...

class BundleState:
    def __init__(self, cameras: Iterable[Optional[Pose]], markers: Markers) -> None: ...
    @property
    def cameras(self) -> List[Optional[Pose]]: ...
    @property
    def markers(self) -> Dict[int, Tuple[NDArray[np.float64], NDArray[np.float64]]]: ...

class Bundle:
    @property
    def state(self) -> BundleState: ...
    @property
    def cameras(self) -> List[Optional[Pose]]: ...
    @property
    def markers(self) -> Dict[int, Tuple[NDArray[np.float64], NDArray[np.float64]]]: ...
    @property
    def cov_markers(self) -> Dict[int, NDArray[np.float64]]: ...
    @property
    def cov_cams(self) -> List[Optional[NDArray[np.float64]]]: ...
    @property
    def info(self) -> NDArray[np.float64]: ...
    @property
    def cov(self) -> Optional[NDArray[np.float64]]: ...
    @property
    def rms_px(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def null_dim(self) -> int: ...
    @property
    def eig_min_ratio(self) -> float: ...
    @property
    def dropped(self) -> int: ...

class Scene:
    def __init__(
        self,
        workspace: ArrayLike,
        candidates: ArrayLike,
        obstacles: Optional[Iterable[Tuple[ArrayLike, float]]] = None,
        *,
        camera: Camera,
        a: float,
        sigma_px: float = 0.3,
        noise_model: NoiseModel = "tangent",
        psi_max_deg: float = 70.0,
        px_min: float = 12.0,
        eps: float = 1e-3,
        sectors: Optional[ArrayLike] = None,
        spacing: float = 0.5,
        z_cam: Optional[ArrayLike] = None,
        headings: Optional[ArrayLike] = None,
    ) -> None: ...
    @property
    def n_candidates(self) -> int: ...
    @property
    def n_cells(self) -> int: ...
    @property
    def candidates(self) -> NDArray[np.float64]: ...

class Placement:
    @property
    def chosen(self) -> NDArray[np.int64]: ...
    @property
    def chosen_candidates(self) -> NDArray[np.float64]: ...
    @property
    def survey_ids(self) -> NDArray[np.int64]: ...
    @property
    def count(self) -> int: ...
    @property
    def feasible_fraction(self) -> float: ...
    @property
    def worst_sigma_pos(self) -> float: ...
    @property
    def cells(self) -> NDArray[np.float64]: ...
    @property
    def sigma(self) -> NDArray[np.float64]: ...
    @property
    def n_vis(self) -> NDArray[np.float64]: ...
    @property
    def sigma_loo(self) -> NDArray[np.float64]: ...
    @property
    def feasible(self) -> NDArray[np.bool_]: ...
    @property
    def margin(self) -> NDArray[np.float64]: ...
    @property
    def method(self) -> Literal["greedy", "ift", "evaluate"]: ...

def observations_from_eqar(markers: Iterable[ObservationLike]) -> List[Observation]: ...
def resection(
    observations: Iterable[ObservationLike],
    markers: Markers,
    camera: Camera,
    a: float,
    *,
    sigma_px: float = 0.3,
    noise_model: NoiseModel = "tangent",
    loss: Loss = "linear",
    huber_delta: float = 1.5,
    dof: Dof = "yaw",
    roll: float = 0.0,
    pitch: float = 0.0,
    x0: Optional[Pose] = None,
) -> Resection: ...
def bundle_adjust(
    frames: Iterable[Iterable[ObservationLike]],
    camera: Camera,
    a: float,
    survey_markers: Mapping[int, Tuple[ArrayLike, float]],
    *,
    rp: Optional[ArrayLike] = None,
    sigma_px: float = 0.3,
    noise_model: NoiseModel = "tangent",
    dof: Dof = "yaw",
    loss: Loss = "linear",
    huber_delta: float = 1.5,
    max_iter: int = 50,
    x0: Optional[BundleState] = None,
) -> Bundle: ...
def init_bundle(
    frames: Iterable[Iterable[ObservationLike]],
    camera: Camera,
    a: float,
    survey_markers: Mapping[int, Tuple[ArrayLike, float]],
    *,
    rp: Optional[ArrayLike] = None,
    sigma_px: float = 0.3,
    noise_model: NoiseModel = "tangent",
    dof: Dof = "yaw",
    refine_rounds: int = 1,
) -> BundleState: ...
def fisher_camera(
    pose: Pose,
    markers: Markers,
    camera: Camera,
    a: float,
    *,
    sigma_px: float = 0.3,
    noise_model: NoiseModel = "tangent",
    dof: Dof = "yaw",
) -> NDArray[np.float64]: ...
def predict_corners(pose: Pose, markers: Markers, camera: Camera, a: float) -> Dict[int, NDArray[np.float64]]: ...
def sigma_pos(info: ArrayLike) -> float: ...
def place_markers(
    scene: Scene,
    tau: float,
    *,
    k_min: int = 2,
    n_max: Optional[int] = None,
    tau_loo: Optional[float] = None,
    sigma_cap: float = 10.0,
    w_k: float = 1.0,
    method: Method = "greedy",
    budget_ms: Optional[int] = None,
    sigma_ts: float = 0.005,
) -> Placement: ...
def evaluate(
    scene: Scene,
    chosen: ArrayLike,
    tau: float,
    *,
    k_min: int = 2,
    tau_loo: Optional[float] = None,
    sigma_cap: float = 10.0,
    w_k: float = 1.0,
) -> Placement: ...
def upright_marker(position: ArrayLike, azimuth: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wall_candidates(
    polygon: ArrayLike,
    spacing: float,
    *,
    z: float = 1.5,
    offset: float = 0.05,
    inward: bool = True,
) -> NDArray[np.float64]: ...
