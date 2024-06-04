import argparse
import json
from pathlib import Path

from openai import OpenAI
from pdf2image import convert_from_path

from common import maybe_download_pdf, run_gpt_4o, to_image_content, url_to_pdf_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("_cache"))
    parser.add_argument('--urls', type=str, required=True, nargs="+")
    parser.add_argument("--not_skip", action="store_true")
    parser.add_argument("--once_shot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI()
    results = []
    if args.once_shot:
        run_fn = run_per_url_once_shot
    else:
        run_fn = run_per_url
    for url in args.urls:
        print(f"Processing {url}")
        results.append(
            run_fn(
                url, client=client, output_dir=args.output_dir,
                cache_dir=args.cache_dir, not_skip=args.not_skip,
            )
        )
    print("done")

    args.output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Saved to {args.output_dir / 'results.json'}")


def run_per_url(url: str, client: OpenAI, output_dir: Path, cache_dir: Path, not_skip: bool):
    pdf_path = url_to_pdf_path(url, cache_dir)
    maybe_download_pdf(url, pdf_path)
    pages = convert_from_path(pdf_path)

    results = []
    for i, page in enumerate(pages, start=1):
        page_output_path = output_dir / f"{pdf_path.name}_{i}.json"
        if not_skip and page_output_path.exists():
            print(f"Skipping {page_output_path}")
            continue

        result = run_gpt_4o(client, [
            {
                "role": "system",
                "content": """The following image is a part of a paper. I would like to understand which sources are being cited and where they are cited in the paper. Please refer to the content listed in the reference section. Output the information in the following json format:
{
   "References": [{"index": str(reference number), "title": str(reference title)}, ...],
   "Citations": [{"ref": [str(reference number), ...], "content": str(citation content)}, ...]
}

"""  # noqa
            },
            {"role": "user", "content": [to_image_content(page, "png")]}
        ], response_format={"type": "json_object"})

        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {result}")

        page_output_path.parent.mkdir(exist_ok=True, parents=True)
        page_output_path.write_text(json.dumps(result, ensure_ascii=False))
        print(f"Saved to {page_output_path}")
        results.append(
            {"pdf": pdf_path.name, "page_index": i, "result": result}
        )

    return results


def run_per_url_once_shot(url: str, client: OpenAI, output_dir: Path, cache_dir: Path, not_skip: bool):
    pdf_path = url_to_pdf_path(url, cache_dir)
    maybe_download_pdf(url, pdf_path)
    pages = convert_from_path(pdf_path)

    result = run_gpt_4o(client, [
        {
            "role": "system",
            "content": """The following image is a part of a paper. I would like to understand which sources are being cited and where they are cited in the paper. Please refer to the content listed in the reference section. Output the information in the following json format:
{
"References": [{"index": str(reference number), "title": str(reference title)}, ...],
"Citations": [{"ref": [str(reference number), ...], "content": str(citation content)}, ...]
}

"""  # noqa
        },
        {"role": "user", "content": [
            to_image_content(page, "png") for page in pages]}
    ], response_format={"type": "json_object"})

    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {result}")

    return result


if __name__ == '__main__':
    main()
