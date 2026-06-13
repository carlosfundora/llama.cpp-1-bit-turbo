use anyhow::Result;
use reqwest::blocking::Client;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

fn main() -> Result<()> {
    let htpplib_version = "refs/tags/v0.40.0";

    let vendor_files = vec![
        (
            "https://github.com/nlohmann/json/releases/latest/download/json.hpp".to_string(),
            "vendor/nlohmann/json.hpp".to_string(),
        ),
        (
            "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp".to_string(),
            "vendor/nlohmann/json_fwd.hpp".to_string(),
        ),
        (
            "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h".to_string(),
            "vendor/stb/stb_image.h".to_string(),
        ),
        (
            "https://github.com/mackron/miniaudio/raw/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h".to_string(),
            "vendor/miniaudio/miniaudio.h".to_string(),
        ),
        (
            format!("https://raw.githubusercontent.com/yhirose/cpp-httplib/{}/httplib.h", htpplib_version),
            "httplib.h".to_string(),
        ),
        (
            format!("https://raw.githubusercontent.com/yhirose/cpp-httplib/{}/LICENSE", htpplib_version),
            "vendor/cpp-httplib/LICENSE".to_string(),
        ),
        (
            "https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h".to_string(),
            "vendor/sheredom/subprocess.h".to_string(),
        ),
    ];

    let client = Client::new();

    for (url, filename) in vendor_files {
        println!("downloading {} to {}", url, filename);
        let path = Path::new(&filename);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let response = client.get(&url).send()?;
        let mut file = File::create(path)?;
        file.write_all(&response.bytes()?)?;
    }

    println!("Splitting httplib.h...");

    let in_file = "httplib.h";
    let h_out = "vendor/cpp-httplib/httplib.h";
    let cc_out = "vendor/cpp-httplib/httplib.cpp";

    let do_split = if Path::new(h_out).exists() {
        let in_time = std::fs::metadata(in_file)?.modified()?;
        let out_time = std::fs::metadata(h_out)?.modified()?;
        in_time > out_time
    } else {
        true
    };

    if do_split {
        let border = "// ----------------------------------------------------------------------------";
        let mut in_implementation = false;

        std::fs::create_dir_all("vendor/cpp-httplib")?;

        let file = File::open(in_file)?;
        let reader = BufReader::new(file);

        let mut fh = File::create(h_out)?;
        let mut fc = File::create(cc_out)?;

        writeln!(fc, "#include \"httplib.h\"")?;
        writeln!(fc, "namespace httplib {{")?;

        for line in reader.lines() {
            let line = line?;
            if line.contains(border) {
                in_implementation = !in_implementation;
            } else if in_implementation {
                let replaced = line.replace("inline ", "");
                writeln!(fc, "{}", replaced)?;
            } else {
                writeln!(fh, "{}", line)?;
            }
        }

        writeln!(fc, "}} // namespace httplib")?;

        println!("Wrote {} and {}", h_out, cc_out);
    } else {
        println!("{} and {} are up to date", h_out, cc_out);
    }

    std::fs::remove_file("httplib.h")?;

    Ok(())
}
