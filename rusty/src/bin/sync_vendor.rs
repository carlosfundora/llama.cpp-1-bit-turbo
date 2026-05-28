use anyhow::{Context, Result};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

const HTTPLIB_VERSION: &str = "refs/tags/v0.40.0";
const BORDER: &str =
    "// ----------------------------------------------------------------------------";

fn main() -> Result<()> {
    let mut vendor_files = vec![
        ("https://github.com/nlohmann/json/releases/latest/download/json.hpp".to_string(), "vendor/nlohmann/json.hpp".to_string()),
        ("https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp".to_string(), "vendor/nlohmann/json_fwd.hpp".to_string()),
        ("https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h".to_string(), "vendor/stb/stb_image.h".to_string()),
        ("https://github.com/mackron/miniaudio/raw/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h".to_string(), "vendor/miniaudio/miniaudio.h".to_string()),
        ("https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h".to_string(), "vendor/sheredom/subprocess.h".to_string()),
    ];

    let httplib_h_url = format!(
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/{}/httplib.h",
        HTTPLIB_VERSION
    );
    let httplib_license_url = format!(
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/{}/LICENSE",
        HTTPLIB_VERSION
    );

    vendor_files.push((httplib_h_url, "httplib.h".to_string()));
    vendor_files.push((
        httplib_license_url,
        "vendor/cpp-httplib/LICENSE".to_string(),
    ));

    // Download files in parallel
    vendor_files
        .par_iter()
        .try_for_each(|(url, filename)| -> Result<()> {
            println!("downloading {} to {}", url, filename);
            if let Some(parent) = Path::new(filename).parent() {
                fs::create_dir_all(parent)?;
            }
            let response = reqwest::blocking::get(url)
                .with_context(|| format!("Failed to download {}", url))?
                .bytes()?;
            fs::write(filename, response)
                .with_context(|| format!("Failed to write {}", filename))?;
            Ok(())
        })?;

    println!("Splitting httplib.h...");
    split_httplib_h("httplib.h", "vendor/cpp-httplib")?;

    fs::remove_file("httplib.h")?;

    Ok(())
}

fn split_httplib_h(in_file: &str, out_dir: &str) -> Result<()> {
    fs::create_dir_all(out_dir)?;

    let h_out_path = format!("{}/httplib.h", out_dir);
    let cc_out_path = format!("{}/httplib.cpp", out_dir);

    let input = File::open(in_file)?;
    let reader = BufReader::new(input);

    let mut h_out = File::create(&h_out_path)?;
    let mut cc_out = File::create(&cc_out_path)?;

    writeln!(cc_out, "#include \"httplib.h\"")?;
    writeln!(cc_out, "namespace httplib {{")?;

    let mut in_implementation = false;

    for line in reader.lines() {
        let line = line?;
        if line.contains(BORDER) {
            in_implementation = !in_implementation;
        } else if in_implementation {
            writeln!(cc_out, "{}", line.replace("inline ", ""))?;
        } else {
            writeln!(h_out, "{}", line)?;
        }
    }

    writeln!(cc_out, "}} // namespace httplib")?;

    println!("Wrote {} and {}", h_out_path, cc_out_path);
    Ok(())
}
