use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fetches the Jinja chat template of a HuggingFace model.", long_about = None)]
struct Args {
    /// The model ID to fetch the chat template for (e.g. microsoft/Phi-3.5-mini-instruct)
    model_id: String,

    /// Optional variant name
    variant: Option<String>,
}

fn fetch_tokenizer_config(model_id: &str) -> Result<String> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/tokenizer_config.json",
        model_id
    );

    let client = reqwest::blocking::Client::builder()
        .user_agent("get_chat_template/1.0")
        .build()?;

    let res = client.get(&url).send()?;

    if res.status() == 401 {
        anyhow::bail!("Access to this model is gated. Note: huggingface-cli login may be required if we supported it natively. For gated models, use the python script or provide an HF_TOKEN.");
    }

    res.error_for_status_ref()?;

    Ok(res.text()?)
}

fn extract_chat_template(config_str: &str, variant: Option<&str>) -> Result<String> {
    let config: Value = match serde_json::from_str(config_str) {
        Ok(v) => v,
        Err(_) => {
            let re = Regex::new(r#"\}([\n\s]*\}[\n\s]*\],[\n\s]*"clean_up_tokenization_spaces")"#)?;
            let fixed_str = re.replace_all(config_str, r#"$1"#);
            serde_json::from_str(&fixed_str).context("Failed to parse tokenizer_config.json even after regex fix")?
        }
    };

    let chat_template = config.get("chat_template").context("chat_template not found in config")?;

    if let Some(template_str) = chat_template.as_str() {
        return Ok(template_str.to_string());
    }

    if let Some(template_arr) = chat_template.as_array() {
        let mut variants = std::collections::HashMap::new();
        for item in template_arr {
            if let (Some(name), Some(template)) = (item.get("name").and_then(|v| v.as_str()), item.get("template").and_then(|v| v.as_str())) {
                variants.insert(name, template);
            }
        }

        let format_variants = || {
            variants.keys().map(|k| format!("\"{}\"", k)).collect::<Vec<_>>().join(", ")
        };

        match variant {
            None => {
                if !variants.contains_key("default") {
                    anyhow::bail!("Please specify a chat template variant (one of {})", format_variants());
                }
                eprintln!("Note: picked \"default\" chat template variant (out of {})", format_variants());
                return Ok(variants.get("default").unwrap().to_string());
            }
            Some(v) => {
                if let Some(template) = variants.get(v) {
                    return Ok(template.to_string());
                } else {
                    anyhow::bail!("Variant {} not found in chat template (found {})", v, format_variants());
                }
            }
        }
    }

    anyhow::bail!("chat_template is neither a string nor a valid array of objects")
}

fn main() -> Result<()> {
    let args = Args::parse();

    let config_str = fetch_tokenizer_config(&args.model_id)?;
    let template = extract_chat_template(&config_str, args.variant.as_deref())?;

    print!("{}", template);

    Ok(())
}
