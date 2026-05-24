use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(serde::Deserialize)]
struct Row {
    backend_name: String,
    op_name: String,
    error_message: String,
    backend_reg_name: String,
    test_mode: String,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let output_filename = if args.len() > 1 {
        &args[1]
    } else {
        "ops.md"
    };

    let ggml_root = std::env::current_dir()?;
    let ops_dir = ggml_root.join("docs").join("ops");

    if !ops_dir.exists() {
        eprintln!("WARNING: ops directory not found: {}", ops_dir.display());
        return Ok(());
    }

    let mut backend_support: BTreeMap<String, BTreeMap<String, Vec<bool>>> = BTreeMap::new();
    let mut all_operations: BTreeSet<String> = BTreeSet::new();
    let mut all_backends: BTreeSet<String> = BTreeSet::new();

    println!("INFO:Parsing GGML operation support files...");
    println!("INFO:Parsing support files from {}...", ops_dir.display());

    for entry in std::fs::read_dir(&ops_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("csv") {
            continue;
        }

        println!("INFO:  Reading: {}", path.file_name().unwrap().to_string_lossy());

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&path)?;

        for result in rdr.deserialize() {
            let record: Row = match result {
                Ok(r) => r,
                Err(_) => continue,
            };

            if record.test_mode != "support" {
                continue;
            }

            let backend_name = record.backend_name.trim().to_string();
            let operation = record.op_name.trim().to_string();
            let supported_str = record.error_message.trim().to_string();
            let backend_reg_name = record.backend_reg_name.trim().to_string();

            if operation.is_empty()
                || backend_name.is_empty()
                || operation == "CONTEXT_ERROR"
                || operation == "BUILD_ERROR"
            {
                continue;
            }

            let is_supported = supported_str.to_lowercase() == "yes";
            let backend_key = if backend_reg_name.is_empty() {
                backend_name
            } else {
                backend_reg_name
            };

            all_backends.insert(backend_key.clone());
            all_operations.insert(operation.clone());

            backend_support
                .entry(backend_key)
                .or_default()
                .entry(operation)
                .or_default()
                .push(is_supported);
        }
    }

    if all_operations.is_empty() {
        eprintln!("ERROR: No operations found. Make sure to run test-backend-ops support --output csv > docs/ops/file.csv first.");
        return Ok(());
    }

    println!(
        "INFO:Found {} operations across {} backends",
        all_operations.len(),
        all_backends.len()
    );

    println!("INFO:Generating markdown...");

    let mut lines = Vec::new();
    lines.push("# GGML Operations".to_string());
    lines.push("".to_string());
    lines.push("List of GGML operations and backend support status.".to_string());
    lines.push("".to_string());
    lines.push("## How to add a backend to this table:".to_string());
    lines.push("".to_string());
    lines.push("1. Run `test-backend-ops support --output csv` with your backend name and redirect output to a csv file in `docs/ops/` (e.g., `docs/ops/CUDA.csv`)".to_string());
    lines.push("2. Regenerate `/docs/ops.md` via `./scripts/create_ops_docs.py`".to_string());
    lines.push("".to_string());
    lines.push("Legend:".to_string());
    lines.push("- ✅ Fully supported by this backend".to_string());
    lines.push("- 🟡 Partially supported by this backend".to_string());
    lines.push("- ❌ Not supported by this backend".to_string());
    lines.push("".to_string());

    let mut header = "| Operation |".to_string();
    for backend in &all_backends {
        header.push_str(&format!(" {} |", backend));
    }

    let mut separator = "|-----------|".to_string();
    for _ in &all_backends {
        separator.push_str("------|");
    }

    lines.push(header);
    lines.push(separator);

    for operation in &all_operations {
        let mut row = format!("| {:>32} |", operation);

        for backend in &all_backends {
            let status = match backend_support.get(backend).and_then(|ops| ops.get(operation)) {
                None => "unsupported",
                Some(support_list) if support_list.is_empty() => "unsupported",
                Some(support_list) => {
                    if support_list.iter().all(|&b| b) {
                        "supported"
                    } else if support_list.iter().any(|&b| b) {
                        "partially supported"
                    } else {
                        "unsupported"
                    }
                }
            };

            let symbol = match status {
                "supported" => "✅",
                "partially supported" => "🟡",
                _ => "❌",
            };

            row.push_str(&format!(" {} |", symbol));
        }
        lines.push(row);
    }

    lines.push("".to_string());

    let docs_dir = ggml_root.join("docs");
    std::fs::create_dir_all(&docs_dir)?;

    let ops_file = docs_dir.join(output_filename);
    std::fs::write(&ops_file, lines.join("\n"))?;

    println!("INFO:Generated: {}", ops_file.display());
    println!("INFO:Operations: {}", all_operations.len());
    println!("INFO:Backends: {}", all_backends.len());

    Ok(())
}
