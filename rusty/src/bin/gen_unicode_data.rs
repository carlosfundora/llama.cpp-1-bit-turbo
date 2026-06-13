use anyhow::Result;
use reqwest::blocking::get;
use unicode_normalization::UnicodeNormalization;

const MAX_CODEPOINTS: u32 = 0x110000;
const UNICODE_DATA_URL: &str = "https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt";

const CODEPOINT_FLAG_UNDEFINED: u16 = 0x0001;
const CODEPOINT_FLAG_NUMBER: u16 = 0x0002;
const CODEPOINT_FLAG_LETTER: u16 = 0x0004;
const CODEPOINT_FLAG_SEPARATOR: u16 = 0x0008;
const CODEPOINT_FLAG_MARK: u16 = 0x0010;
const CODEPOINT_FLAG_PUNCTUATION: u16 = 0x0020;
const CODEPOINT_FLAG_SYMBOL: u16 = 0x0040;
const CODEPOINT_FLAG_CONTROL: u16 = 0x0080;

fn get_flag_for_category(categ: &str) -> u16 {
    match categ {
        "Cn" => CODEPOINT_FLAG_UNDEFINED,
        "Cc" | "Cf" | "Co" | "Cs" => CODEPOINT_FLAG_CONTROL,
        "Ll" | "Lm" | "Lo" | "Lt" | "Lu" | "L&" => CODEPOINT_FLAG_LETTER,
        "Mc" | "Me" | "Mn" => CODEPOINT_FLAG_MARK,
        "Nd" | "Nl" | "No" => CODEPOINT_FLAG_NUMBER,
        "Pc" | "Pd" | "Pe" | "Pf" | "Pi" | "Po" | "Ps" => CODEPOINT_FLAG_PUNCTUATION,
        "Sc" | "Sk" | "Sm" | "So" => CODEPOINT_FLAG_SYMBOL,
        "Zl" | "Zp" | "Zs" => CODEPOINT_FLAG_SEPARATOR,
        _ => CODEPOINT_FLAG_UNDEFINED,
    }
}

fn parse_hex(s: &str) -> u32 {
    if s.is_empty() {
        0
    } else {
        u32::from_str_radix(s, 16).unwrap_or(0)
    }
}

fn main() -> Result<()> {
    let res = get(UNICODE_DATA_URL)?;
    let text = res.text()?;

    let mut codepoint_flags = vec![CODEPOINT_FLAG_UNDEFINED; MAX_CODEPOINTS as usize];
    let mut table_lowercase = Vec::new();
    let mut table_uppercase = Vec::new();
    let mut table_nfd: Vec<(u32, u32)> = Vec::new();

    let mut prev: Option<(u32, u32, u32, String, String)> = None;

    let mut process_cpt = |cpt: u32, cpt_lower: u32, cpt_upper: u32, categ: &str, _bidir: &str| {
        let flag = get_flag_for_category(categ);
        if (cpt as usize) < codepoint_flags.len() {
            codepoint_flags[cpt as usize] = flag;
        }

        if cpt_lower != 0 {
            table_lowercase.push((cpt, cpt_lower));
        }

        if cpt_upper != 0 {
            table_uppercase.push((cpt, cpt_upper));
        }

        if let Some(ch) = std::char::from_u32(cpt) {
            let nfd: String = ch.nfd().collect();
            if let Some(first_char) = nfd.chars().next() {
                let norm = first_char as u32;
                if cpt != norm {
                    table_nfd.push((cpt, norm));
                }
            }
        }
    };

    for line in text.lines() {
        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() < 14 {
            continue;
        }

        let cpt = parse_hex(parts[0]);
        let cpt_lower = parse_hex(parts[13]);
        let cpt_upper = parse_hex(parts[12]);
        let categ = parts[2].trim().to_string();
        let bidir = parts[4].trim().to_string();
        let name = parts[1];

        if name.ends_with(", First>") {
            prev = Some((cpt, cpt_lower, cpt_upper, categ, bidir));
            continue;
        }

        if name.ends_with(", Last>") {
            if let Some(p) = &prev {
                assert_eq!(p.1, 0);
                assert_eq!(p.2, 0);
                assert_eq!(p.3, categ);
                assert_eq!(p.4, bidir);
                for c in p.0..cpt {
                    process_cpt(c, cpt_lower, cpt_upper, &categ, &bidir);
                }
            }
        }

        process_cpt(cpt, cpt_lower, cpt_upper, &categ, &bidir);
    }

    let mut table_whitespace = Vec::new();
    table_whitespace.extend(0x0009..=0x000D);
    table_whitespace.extend(0x2000..=0x200A);
    table_whitespace.extend_from_slice(&[
        0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
    ]);

    table_whitespace.sort_unstable();
    table_lowercase.sort_unstable();
    table_uppercase.sort_unstable();
    table_nfd.sort_unstable();

    let mut ranges_flags = vec![(0u32, codepoint_flags[0])];
    for (codepoint, &flags) in codepoint_flags.iter().enumerate() {
        let codepoint = codepoint as u32;
        if flags != ranges_flags.last().unwrap().1 {
            ranges_flags.push((codepoint, flags));
        }
    }
    ranges_flags.push((MAX_CODEPOINTS, 0x0000));

    let mut ranges_nfd = vec![(0u32, 0u32, 0u32)];
    for &(codepoint, norm) in &table_nfd {
        let mut start = ranges_nfd.last().unwrap().0;
        if ranges_nfd.last().unwrap().clone() != (start, codepoint.saturating_sub(1), norm) {
            ranges_nfd.push((0, 0, 0));
            start = codepoint;
        }
        let last_idx = ranges_nfd.len() - 1;
        ranges_nfd[last_idx] = (start, codepoint, norm);
    }

    println!("// generated with scripts/gen-unicode-data.py");
    println!();
    println!("#include \"unicode-data.h\"");
    println!();
    println!("#include <cstdint>");
    println!("#include <vector>");
    println!("#include <unordered_map>");
    println!("#include <unordered_set>");
    println!();

    println!("const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {{  // start, flags // last=next_start-1");
    for (start, flags) in ranges_flags {
        println!("{{0x{:06X}, 0x{:04X}}},", start, flags);
    }
    println!("}};\n");

    println!("const std::unordered_set<uint32_t> unicode_set_whitespace = {{");
    for cp in table_whitespace {
        println!("0x{:06X},", cp);
    }
    println!("}};\n");

    println!("const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {{");
    for (cpt, c_lower) in table_lowercase {
        println!("{{0x{:06X}, 0x{:06X}}},", cpt, c_lower);
    }
    println!("}};\n");

    println!("const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {{");
    for (cpt, c_upper) in table_uppercase {
        println!("{{0x{:06X}, 0x{:06X}}},", cpt, c_upper);
    }
    println!("}};\n");

    println!("const std::vector<range_nfd> unicode_ranges_nfd = {{  // start, last, nfd");
    for (start, last, nfd) in ranges_nfd {
        println!("{{0x{:06X}, 0x{:06X}, 0x{:06X}}},", start, last, nfd);
    }
    println!("}};");

    Ok(())
}
