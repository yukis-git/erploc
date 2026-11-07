//! Python bindings for erploc: equirectangular (360 degree) camera localisation and marker placement.
//!
//! The native module is `erploc._erploc`; the `erploc` package re-exports it.
//!
//! Threading: every computation releases the GIL. Inputs are converted into owned Rust values first, so no Python object or
//! numpy buffer is touched while the GIL is released. Errors: invalid input raises `ValueError` (or `TypeError` for a wrong
//! kind of object), a failed computation raises `erploc.NumericalError`; the library never panics on input that reaches it
//! (a panic that did occur would surface as `RuntimeError`).

mod conv;
mod est;
mod place;

use pyo3::prelude::*;

/// Converts eqar detections (`eqar.Marker` objects or their `to_dict()` dicts) into `Observation`s. No eqar import.
#[pyfunction]
fn observations_from_eqar<'py>(py: Python<'py>, markers: &Bound<'py, PyAny>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    conv::observations_from_eqar(py, markers)
}

type MarkerPose<'py> = (Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray2<f64>>);

/// Pose `(position (3,), rotation (3, 3))` of an upright marker (vertical, normal `(cos az, sin az, 0)`, y axis up): the pose
/// that `place_markers` assumes for a candidate `[x, y, z, azimuth]`.
#[pyfunction]
fn upright_marker<'py>(py: Python<'py>, position: &Bound<'py, PyAny>, azimuth: f64) -> PyResult<MarkerPose<'py>> {
    let p = conv::arr3(&conv::f64s(position, "position", &[3])?);
    let p = erploc_cli::io::v3(&p, "position").map_err(conv::cli_err)?;
    let az = erploc_cli::io::finite(azimuth, "azimuth").map_err(conv::cli_err)?;
    let mk = erploc_core::markers::upright_pose(0, &p, az);
    Ok((conv::arr1(py, p.to_vec()), conv::m3_py(py, &mk.r)?))
}

#[pymodule]
fn _erploc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NumericalError", m.py().get_type::<conv::NumericalError>())?;
    m.add_class::<conv::Camera>()?;
    m.add_class::<conv::Pose>()?;
    m.add_class::<conv::Observation>()?;
    m.add_class::<est::Resection>()?;
    m.add_class::<est::Bundle>()?;
    m.add_class::<est::BundleState>()?;
    m.add_class::<place::Scene>()?;
    m.add_class::<place::Placement>()?;
    m.add_function(wrap_pyfunction!(observations_from_eqar, m)?)?;
    m.add_function(wrap_pyfunction!(upright_marker, m)?)?;
    m.add_function(wrap_pyfunction!(est::resection, m)?)?;
    m.add_function(wrap_pyfunction!(est::bundle_adjust, m)?)?;
    m.add_function(wrap_pyfunction!(est::init_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(est::fisher_camera, m)?)?;
    m.add_function(wrap_pyfunction!(est::sigma_pos, m)?)?;
    m.add_function(wrap_pyfunction!(est::predict_corners, m)?)?;
    m.add_function(wrap_pyfunction!(place::place_markers, m)?)?;
    m.add_function(wrap_pyfunction!(place::evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(place::wall_candidates, m)?)?;
    Ok(())
}
