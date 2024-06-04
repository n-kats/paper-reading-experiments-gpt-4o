import argparse
import json
from pathlib import Path

import polars as pl
import tiktoken

from common import maybe_download_pdf, read_pdf_by_pypdf, url_to_pdf_path


def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def count_non_empty_chars(text: str) -> int:
    return len(text.replace(" ", "").replace("\n", ""))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("_cache"))
    parser.add_argument('--urls', type=str, required=True, nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()

    details = []
    for url in args.urls:
        pdf_path = url_to_pdf_path(url, args.cache_dir)
        maybe_download_pdf(url, pdf_path)
        texts = read_pdf_by_pypdf(pdf_path)
        for i, text in enumerate(texts, start=1):
            details.append({
                "pdf": pdf_path.name, "page": i,
                "text": text,
                "length": len(text),
                "non_empty_chars": count_non_empty_chars(text),
                "o200k_base(gpt-4o)": count_tokens(text, "gpt-4o"),
                "cl100k_base(gpt-4,gpt-3.5)": count_tokens(text, "gpt-4"),
            })

    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_detail_jsonl = args.output_dir / "detail.jsonl"
    output_detail_csv = args.output_dir / "detail.csv"
    output_grouped_csv = args.output_dir / "grouped.csv"

    with output_detail_jsonl.open("w") as f_out:
        for per_page in details:
            print(json.dumps(per_page, ensure_ascii=False), file=f_out)

    without_text = [{k: v for k, v in d.items() if k != "text"}
                    for d in details]
    df = pl.DataFrame(without_text)
    df.write_csv(output_detail_csv)
    print("# result(per_page):")
    print(df)
    print("This is saved to", output_detail_csv)
    print()

    df_grouped = df.group_by("pdf").agg([
        pl.col("page").max().alias("page_count"),
        pl.col("length").sum().alias("length(sum)"),
        pl.col("length").mean().alias("length(mean)"),
        pl.col("non_empty_chars").sum().alias("non_empty_chars(sum)"),
        pl.col("non_empty_chars").mean().alias("non_empty_chars(mean)"),
        pl.col("o200k_base(gpt-4o)").sum().alias("o200k_base(gpt-4o)(sum)"),
        pl.col("o200k_base(gpt-4o)").mean().alias("o200k_base(gpt-4o)(mean)"),
        pl.col(
            "cl100k_base(gpt-4,gpt-3.5)").sum().alias("cl100k_base(gpt-4,gpt-3.5)(sum)"),
        pl.col(
            "cl100k_base(gpt-4,gpt-3.5)").mean().alias("cl100k_base(gpt-4,gpt-3.5)(mean)"),
    ])
    total_sum = df.sum().to_dict()
    total_mean = df.mean().to_dict()
    total = pl.DataFrame([
        {
            "pdf": "TOTAL",
            "page_count": len(details),
            "length(sum)": total_sum["length"][0],
            "length(mean)": total_mean["length"][0],
            "non_empty_chars(sum)": total_sum["non_empty_chars"][0],
            "non_empty_chars(mean)": total_mean["non_empty_chars"][0],
            "o200k_base(gpt-4o)(sum)": total_sum["o200k_base(gpt-4o)"][0],
            "o200k_base(gpt-4o)(mean)": total_mean["o200k_base(gpt-4o)"][0],
            "cl100k_base(gpt-4,gpt-3.5)(sum)": total_sum["cl100k_base(gpt-4,gpt-3.5)"][0],
            "cl100k_base(gpt-4,gpt-3.5)(mean)": total_mean["cl100k_base(gpt-4,gpt-3.5)"][0],
        }
    ])

    df_grouped = pl.concat([df_grouped, total])
    df_grouped.write_csv(output_grouped_csv)

    print("# result(grouped):")
    print(df_grouped)
    print("This is saved to", output_grouped_csv)


if __name__ == '__main__':
    main()
