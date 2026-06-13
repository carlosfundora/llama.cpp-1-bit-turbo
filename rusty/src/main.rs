use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Parser, Debug)]
#[command(author, version, about = "Verify model checksums", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "SHA256SUMS")]
    hash_file: PathBuf,

    #[arg(short, long, default_value = ".")]
    base_dir: PathBuf,
}

fn sha256sum(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 16 * 1024 * 1024]; // 16 MB chunks

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(hex::encode(hasher.finalize()))
}

struct VerificationResult {
    filename: String,
    valid_checksum: String,
    file_missing: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let hash_list_file = args.base_dir.join(&args.hash_file);
    if !hash_list_file.exists() {
        eprintln!("Hash list file not found: {}", hash_list_file.display());
        std::process::exit(1);
    }

    let file = File::open(&hash_list_file).context("Failed to open hash list file")?;
    let reader = BufReader::new(file);

    let mut tasks = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split("  ").collect();
        if parts.len() != 2 {
            continue;
        }

        let expected_hash = parts[0].to_string();
        let filename = parts[1].to_string();

        tasks.push((expected_hash, filename));
    }

    println!(
        "{:<40} {:^20} {:^20}",
        "filename", "valid checksum", "file missing"
    );
    println!("{:-<80}", "");

    // Using Mutex to serialize console output nicely, though eprintln could interleave
    let output_lock = Mutex::new(());

    let results: Vec<VerificationResult> = tasks
        .into_par_iter()
        .map(|(expected_hash, filename)| {
            let file_path = args.base_dir.join(&filename);

            {
                let _lock = output_lock.lock().unwrap();
                eprintln!("Verifying the checksum of {}", file_path.display());
            }

            let (valid_checksum, file_missing) = if file_path.exists() {
                match sha256sum(&file_path) {
                    Ok(actual_hash) => {
                        if actual_hash == expected_hash {
                            ("V", "")
                        } else {
                            ("", "")
                        }
                    }
                    Err(_) => ("", ""),
                }
            } else {
                ("", "X")
            };

            VerificationResult {
                filename,
                valid_checksum: valid_checksum.to_string(),
                file_missing: file_missing.to_string(),
            }
        })
        .collect();

    for result in results {
        println!(
            "{:<40} {:^20} {:^20}",
            result.filename, result.valid_checksum, result.file_missing
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_sha256sum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"hello world").unwrap();

        let hash = sha256sum(temp_file.path()).unwrap();
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_sha256sum_empty() {
        let temp_file = NamedTempFile::new().unwrap();

        let hash = sha256sum(temp_file.path()).unwrap();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
