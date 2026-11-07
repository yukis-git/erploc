//! erploc-cli: command-line front end of the erploc Rust port (the only crate with serde).
//!
//! Subcommands `place`, `resect`, `ba`, `site` read JSON, validate everything in [`io`], call the library and write JSON (and a CSV
//! for `place`). The `cmd_*` functions are pure over strings (the `site` command reads its two table files), so tests need no
//! process spawn. `e2e` runs the end-to-end checks E2E-1..4 ([`e2e`]). Exit codes: see [`error::CliError::exit_code`].
pub mod cli;
pub mod cmd_ba;
pub mod cmd_place;
pub mod cmd_resect;
pub mod cmd_site;
pub mod e2e;
pub mod error;
pub mod io;

pub use cmd_ba::run_ba;
pub use cmd_place::{run_place, Method, PlaceOpts, PlaceOut};
pub use cmd_resect::run_resect;
pub use cmd_site::run_site;
pub use error::CliError;
