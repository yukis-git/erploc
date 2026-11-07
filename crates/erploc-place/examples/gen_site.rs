//! Writes the fully synthetic site table (format `ERPSITE1`, see `docs/site-tables.md`) to the path given as the first argument.
//!
//! `cargo run --release -p erploc-place --example gen_site -- site.bin`
fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: gen_site OUT.bin");
        std::process::exit(2);
    };
    let bytes = erploc_place::site_synth::synthetic_site_bytes().expect("synthetic table");
    std::fs::write(&path, &bytes).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    println!("wrote {path} ({} bytes)", bytes.len());
}
