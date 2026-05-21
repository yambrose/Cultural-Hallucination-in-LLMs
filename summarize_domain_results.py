import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


DOMAIN_COLUMNS = [
    "culture",
    "dataset_type",
    "language",
    "domain",
    "model",
    "judge_model",
    "total",
    "refusals",
    "non_refusals",
    "hallucinations",
    "hallucinations_not_refused",
    "correct",
    "refusal_rate",
    "hallucination_rate_not_refused",
    "hallucination_rate_all",
    "correct_rate",
]

CULTURE_COLUMNS = [
    "culture",
    "dataset_type",
    "canonical_domain",
    "total",
    "refusals",
    "non_refusals",
    "hallucinations",
    "hallucinations_not_refused",
    "correct",
    "refusal_rate",
    "hallucination_rate_not_refused",
    "hallucination_rate_all",
    "correct_rate",
]

CULTURES = {
    "model_outputs_gujarati": {
        "culture": "gujarati",
        "judge_dir": "guj",
        "dataset_prefix": "Guj",
        "languages": {"en": "EN", "gu": "GU", "hi": "HI"},
    },
    "model_outputs_tamil": {
        "culture": "tamil",
        "judge_dir": "tam",
        "dataset_prefix": "Tam",
        "languages": {"en": "EN", "ta": "TA", "hi": "HI"},
    },
}

CANONICAL_DOMAINS = {
    "Food & Cuisine": {
        "food & cuisine",
        "भोजन एवं व्यंजन",
        "ખોરાક અને ભોજન",
        "உணவு & சமையல்",
    },
    "Geography": {
        "geography",
        "भूगोल",
        "ભૂગોળ",
        "புவியியல்",
    },
    "History": {
        "history",
        "इतिहास",
        "ઈતિહાસ",
        "வரலாறு",
    },
    "Festivals": {
        "festivals",
        "समारोह",
        "તહેવારો",
        "திருவிழாக்கள்",
    },
}

DOMAIN_TO_CANONICAL = {
    variant.casefold(): canonical
    for canonical, variants in CANONICAL_DOMAINS.items()
    for variant in variants
}


@dataclass
class BuildResult:
    domain_rows: list[dict]
    culture_rows: list[dict]
    skipped_pairs: list[dict]


def normalize_dataset_type(value):
    value = (value or "").lower()
    if "nonexistent" in value:
        return "nonexistent"
    if "precise" in value:
        return "precise"
    return value or "unknown"


def normalize_cell(value):
    return "" if value is None else str(value).strip()


def canonical_domain(domain):
    normalized = normalize_cell(domain)
    return DOMAIN_TO_CANONICAL.get(normalized.casefold(), normalized or "unknown")


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_judge(path):
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    details = payload.get("details") or []
    if not details:
        return None
    return {
        "path": path,
        "judge_model": payload.get("judge_model", ""),
        "details": details,
    }


def row_key(row):
    return (
        normalize_cell(row.get("id")),
        normalize_cell(row.get("question")),
        normalize_cell(row.get("gold_answer")),
        normalize_cell(row.get("model_answer")),
    )


def load_judge_candidates(judge_dir):
    if not judge_dir.exists():
        return []

    candidates = []
    for path in sorted(judge_dir.glob("*.json")):
        if "partial" in path.name:
            continue
        candidate = read_judge(path)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def find_matching_judge(output_rows, candidates):
    output_keys = {row_key(row) for row in output_rows}
    best = None
    best_matches = 0

    for candidate in candidates:
        details = candidate["details"]
        if len(details) != len(output_rows):
            continue

        judge_keys = {row_key(row) for row in details}
        matches = len(output_keys & judge_keys)
        if matches > best_matches:
            best = candidate
            best_matches = matches

    if best is None or best_matches != len(output_rows):
        return None
    return best


def read_xlsx_rows(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", ns))
                )

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else value_node.text or ""
                if cell.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                values.append(value)
            rows.append(values)

    if not rows:
        return []

    headers = [normalize_cell(cell) for cell in rows[0]]
    return [
        {headers[index]: row[index] if index < len(row) else "" for index in range(len(headers))}
        for row in rows[1:]
    ]


def find_column(columns, options):
    lowered = {column.lower(): column for column in columns}
    for option in options:
        found = lowered.get(option.lower())
        if found:
            return found
    return None


def load_domain_lookup(root, culture_info, dataset_type, language):
    lang_suffix = culture_info["languages"].get(language)
    if not lang_suffix:
        return {}

    dataset_kind = "IndicWikiQA" if dataset_type == "precise" else "IndicNonExi"
    path = root / "datasets" / f"{dataset_kind}-{culture_info['dataset_prefix']}_{lang_suffix}.xlsx"
    if not path.exists():
        return {}

    rows = read_xlsx_rows(path)
    if not rows:
        return {}

    columns = list(rows[0].keys())
    question_col = find_column(
        columns,
        ["question", "questions", "Question", "Questions", "Question_Gujarati"],
    )
    domain_col = find_column(
        columns,
        ["domain", "Domain", "Domain_Gujarati"],
    )
    if not question_col or not domain_col:
        return {}

    return {
        normalize_cell(row.get(question_col)): normalize_cell(row.get(domain_col))
        for row in rows
        if normalize_cell(row.get(question_col))
    }


def metric_row(group_key, rows):
    total = len(rows)
    refusals = sum(1 for row in rows if row["is_abstaining"])
    non_refusals = total - refusals
    hallucinations = sum(1 for row in rows if row["is_hallucinated"])
    hallucinations_not_refused = sum(
        1 for row in rows if row["is_hallucinated"] and not row["is_abstaining"]
    )
    correct = sum(1 for row in rows if not row["is_hallucinated"])

    result = dict(group_key)
    result.update(
        {
            "total": total,
            "refusals": refusals,
            "non_refusals": non_refusals,
            "hallucinations": hallucinations,
            "hallucinations_not_refused": hallucinations_not_refused,
            "correct": correct,
            "refusal_rate": refusals / total if total else 0.0,
            "hallucination_rate_not_refused": hallucinations_not_refused / non_refusals
            if non_refusals
            else 0.0,
            "hallucination_rate_all": hallucinations / total if total else 0.0,
            "correct_rate": correct / total if total else 0.0,
        }
    )
    return result


def build_domain_rows(example_rows):
    domain_groups = defaultdict(list)
    for row in example_rows:
        domain_key = {
            "culture": row["culture"],
            "dataset_type": row["dataset_type"],
            "language": row["language"],
            "domain": row["domain"],
            "model": row["model"],
            "judge_model": row["judge_model"],
        }
        domain_groups[tuple(domain_key.items())].append(row)

    return [metric_row(dict(key), rows) for key, rows in sorted(domain_groups.items())]


def build_culture_rows(example_rows):
    culture_groups = defaultdict(list)
    for row in example_rows:
        culture_key = {
            "culture": row["culture"],
            "dataset_type": row["dataset_type"],
            "canonical_domain": canonical_domain(row["domain"]),
        }
        culture_groups[tuple(culture_key.items())].append(row)

    return [metric_row(dict(key), rows) for key, rows in sorted(culture_groups.items())]


def build_tables(root):
    root = Path(root)
    example_rows = []
    skipped_pairs = []
    domain_lookup_cache = {}

    for output_root_name, culture_info in CULTURES.items():
        output_root = root / output_root_name
        if not output_root.exists():
            continue

        for output_path in sorted(output_root.glob("*/*/*.jsonl")):
            output_rows = read_jsonl(output_path)
            if not output_rows:
                skipped_pairs.append(
                    {"output_file": str(output_path), "reason": "empty output file"}
                )
                continue

            language = output_path.parent.name
            dataset_type = normalize_dataset_type(
                output_rows[0].get("dataset_type") or output_path.parent.parent.name
            )
            model = output_rows[0].get("model_tag") or output_path.stem
            judge_dir = root / "judge_results" / culture_info["judge_dir"] / language
            candidates = load_judge_candidates(judge_dir)
            judge = find_matching_judge(output_rows, candidates)
            if judge is None:
                skipped_pairs.append(
                    {
                        "output_file": str(output_path),
                        "judge_dir": str(judge_dir),
                        "reason": "no matching judge file with identical evaluated rows",
                    }
                )
                continue

            lookup_key = (culture_info["culture"], dataset_type, language)
            if lookup_key not in domain_lookup_cache:
                domain_lookup_cache[lookup_key] = load_domain_lookup(
                    root, culture_info, dataset_type, language
                )
            domain_lookup = domain_lookup_cache[lookup_key]
            judge_by_key = {row_key(row): row for row in judge["details"]}

            for output_row in output_rows:
                judge_row = judge_by_key[row_key(output_row)]
                question = normalize_cell(output_row.get("question"))
                domain = normalize_cell(output_row.get("domain"))
                if not domain:
                    domain = domain_lookup.get(question, "")
                if not domain:
                    domain = "unknown"

                example_rows.append(
                    {
                        "culture": culture_info["culture"],
                        "dataset_type": dataset_type,
                        "language": language,
                        "domain": domain,
                        "model": model,
                        "judge_model": judge["judge_model"],
                        "is_abstaining": bool(judge_row.get("is_abstaining")),
                        "is_hallucinated": bool(judge_row.get("is_hallucinated")),
                    }
                )

    domain_rows = build_domain_rows(example_rows)
    culture_rows = build_culture_rows(example_rows)
    return BuildResult(domain_rows, culture_rows, skipped_pairs)


def format_csv_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_csv_value(row.get(column, "")) for column in columns})


def main():
    parser = argparse.ArgumentParser(
        description="Summarize domain-specific judge results by culture, language, domain, and dataset type."
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--out_dir",
        default="tables",
        help="Directory where CSV summaries will be written.",
    )
    args = parser.parse_args()

    result = build_tables(Path(args.root))
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "domain_specific_results.csv", result.domain_rows, DOMAIN_COLUMNS)
    write_csv(out_dir / "culture_combined_results.csv", result.culture_rows, CULTURE_COLUMNS)
    write_csv(
        out_dir / "skipped_pairs.csv",
        result.skipped_pairs,
        ["output_file", "judge_dir", "reason"],
    )

    print(f"Wrote {len(result.domain_rows)} domain rows to {out_dir / 'domain_specific_results.csv'}")
    print(f"Wrote {len(result.culture_rows)} culture rows to {out_dir / 'culture_combined_results.csv'}")
    print(f"Wrote {len(result.skipped_pairs)} skipped pairs to {out_dir / 'skipped_pairs.csv'}")


if __name__ == "__main__":
    main()
