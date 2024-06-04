import argparse
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
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI()
    for url in args.urls:
        print(f"Processing {url}")
        run_per_url(url, client=client, output_dir=args.output_dir,
                    cache_dir=args.cache_dir, not_skip=args.not_skip)


def run_per_url(url: str, client: OpenAI, output_dir: Path, cache_dir: Path, not_skip: bool):
    pdf_path = url_to_pdf_path(url, cache_dir)
    maybe_download_pdf(url, pdf_path)
    pages = convert_from_path(pdf_path)
    for i, page in enumerate(pages, start=1):
        page_output_path = output_dir / f"{pdf_path.name}_{i}.json"
        if not_skip and page_output_path.exists():
            print(f"Skipping {page_output_path}")
            continue

        result = run_gpt_4o(client, [
            {
                "role": "system",
                "content": "This is a part of a paper. Determine whether this section indicates that the proposed method in this paper is superior to GPT-4o. If GPT-4o is not mentioned, set is_superior_to_gpt_4o to null. Provide the reason in the following JSON format: {\"is_superior_to_gpt_4o\": bool | null, \"reason\": string}"  # noqa
            },
            {"role": "user", "content": [to_image_content(page, "png")]}
        ], response_format={"type": "json_object"})

        page_output_path.parent.mkdir(exist_ok=True, parents=True)
        page_output_path.write_text(result)
        print(f"Saved to {page_output_path}")


if __name__ == '__main__':
    main()
