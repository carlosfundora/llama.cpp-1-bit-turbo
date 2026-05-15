use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "compare-logprobs",
    about = "Compare logits from JSON log files"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compare {
        input1: PathBuf,
        input2: PathBuf,
        output: PathBuf,
    },
}

#[derive(Deserialize, Debug)]
struct LogprobEntry {
    __index: Option<i64>,
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    logprobs: Logprobs,
}

#[derive(Deserialize, Debug)]
struct Logprobs {
    content: Option<Vec<ContentEntry>>,
    tokens: Option<Vec<String>>,
    token_logprobs: Option<Vec<f64>>,
}

#[derive(Deserialize, Debug)]
struct ContentEntry {
    top_logprobs: Vec<TopLogprob>,
}

#[derive(Deserialize, Debug)]
struct TopLogprob {
    token: String,
    logprob: f64,
}

fn get_token_logprobs(data: &LogprobEntry) -> Option<(String, f64)> {
    if let Some(choices) = data.choices.first() {
        if let Some(content) = &choices.logprobs.content {
            if let Some(first_content) = content.first() {
                if let Some(top) = first_content.top_logprobs.first() {
                    return Some((top.token.clone(), top.logprob));
                }
            }
        } else if let (Some(tokens), Some(token_logprobs)) =
            (&choices.logprobs.tokens, &choices.logprobs.token_logprobs)
        {
            if let (Some(token), Some(&logprob)) = (tokens.first(), token_logprobs.first()) {
                return Some((token.clone(), logprob));
            }
        }
    }
    None
}

fn clean_text(text: &str) -> String {
    format!(
        "'{}'",
        text.replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("\r", "\\r")
            .replace("|", "\\|")
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compare {
            input1,
            input2,
            output,
        } => {
            let f1 = File::open(&input1).context("Failed to open input1")?;
            let f2 = File::open(&input2).context("Failed to open input2")?;
            let mut fout = File::create(&output).context("Failed to create output")?;

            let reader1 = BufReader::new(f1);
            let reader2 = BufReader::new(f2);

            let lines1: Vec<String> = reader1.lines().collect::<Result<_, _>>()?;
            let lines2: Vec<String> = reader2.lines().collect::<Result<_, _>>()?;

            anyhow::ensure!(
                lines1.len() == lines2.len(),
                "Input files must have the same number of lines: {} vs {}",
                lines1.len(),
                lines2.len()
            );

            let name1 = input1.file_name().unwrap().to_string_lossy().to_string();
            let name2 = input2.file_name().unwrap().to_string_lossy().to_string();

            let tab_header = [
                "idx".to_string(),
                name1,
                "logprob_1".to_string(),
                name2,
                "logprob_2".to_string(),
                "diff (abs)".to_string(),
            ];

            let mut tab_entries: Vec<Vec<String>> = Vec::with_capacity(lines1.len());

            for (i, (line1, line2)) in lines1.iter().zip(lines2.iter()).enumerate() {
                if line1.trim().is_empty() || line2.trim().is_empty() {
                    continue;
                }

                let data1: LogprobEntry = serde_json::from_str(line1)?;
                let data2: LogprobEntry = serde_json::from_str(line2)?;

                let idx1 = data1.__index.unwrap_or(-1);
                let idx2 = data2.__index.unwrap_or(-1);

                if idx1 != idx2 {
                    eprintln!(
                        "Warning: Mismatched indices at line {}: {} vs {}",
                        i, idx1, idx2
                    );
                }

                let (token1, logprob1) =
                    get_token_logprobs(&data1).unwrap_or(("".to_string(), 0.0));
                let (token2, logprob2) =
                    get_token_logprobs(&data2).unwrap_or(("".to_string(), 0.0));

                let token1_clean = clean_text(&token1);
                let token2_clean = clean_text(&token2);
                let abs_diff = (logprob1 - logprob2).abs();

                tab_entries.push(vec![
                    (idx1 + 1).to_string(),
                    token1_clean,
                    format!("{:.4}", logprob1),
                    token2_clean,
                    format!("{:.4}", logprob2),
                    format!("{:.4}", abs_diff),
                ]);
            }

            let mut tab_max_widths: Vec<usize> = tab_header.iter().map(|h| h.len()).collect();

            for entry in &tab_entries {
                for (j, col) in entry.iter().enumerate() {
                    if col.len() > tab_max_widths[j] {
                        tab_max_widths[j] = col.len();
                    }
                }
            }

            writeln!(fout, "# Logits Comparison Report\n")?;

            let mut output_str = String::new();
            for (j, h) in tab_header.iter().enumerate() {
                output_str.push_str(&format!("| {:width$} ", h, width = tab_max_widths[j]));
            }
            output_str.push_str("|\n");

            for (j, _) in tab_header.iter().enumerate() {
                output_str.push_str(&format!("|{}", "-".repeat(tab_max_widths[j] + 2)));
            }
            output_str.push_str("|\n");

            for entry in &tab_entries {
                for (j, col) in entry.iter().enumerate() {
                    output_str.push_str(&format!("| {:width$} ", col, width = tab_max_widths[j]));
                }
                output_str.push_str("|\n");
            }

            write!(fout, "{}", output_str)?;
            println!("{}", output_str);
            println!("Report written to {}", output.display());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        assert_eq!(clean_text("hello"), "'hello'");
        assert_eq!(clean_text("hello\nworld"), "'hello\\nworld'");
    }

    #[test]
    fn test_get_token_logprobs_llama() {
        let json = r#"{
            "choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "hello", "logprob": -0.1}]}]}}],
            "__index": 0
        }"#;
        let data: LogprobEntry = serde_json::from_str(json).unwrap();
        assert_eq!(get_token_logprobs(&data), Some(("hello".to_string(), -0.1)));
    }

    #[test]
    fn test_get_token_logprobs_vllm() {
        let json = r#"{
            "choices": [{"logprobs": {"tokens": ["hello"], "token_logprobs": [-0.15]}}],
            "__index": 0
        }"#;
        let data: LogprobEntry = serde_json::from_str(json).unwrap();
        assert_eq!(get_token_logprobs(&data), Some(("hello".to_string(), -0.15)));
    }
}
