import base64
from io import BytesIO
from pathlib import Path

import httpx
import pypdf
from PIL import Image


def url_to_pdf_path(url: str, output_dir: Path) -> Path:
    return maybe_add_pdf_suffix(output_dir / Path(url).name)

def maybe_add_pdf_suffix(path: Path) -> Path:
    if path.suffix == ".pdf":
        return path
    return path.parent / (path.name + ".pdf")

def maybe_download_pdf(url: str, output_path: Path, verbose: bool = True):
    if output_path.exists():
        if verbose:
            print(f"Already downloaded {url} to {output_path}")
        return

    if verbose:
        print(f"Downloading {url} to {output_path}")

    response = httpx.get(url)
    response.raise_for_status()

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_bytes(response.content)

    if verbose:
        print(f"Downloaded {url} to {output_path}")



def to_image_content(image: Image, image_type: str):
    with BytesIO() as f_out:
        image.save(f_out, format=image_type)
        encoded = base64.b64encode(f_out.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{image_type};base64,{encoded}"},
    }


def run_gpt_4o(client, messages, **kwargs):
    return (
        client.chat.completions.create(model="gpt-4o", messages=messages, **kwargs)
        .choices[0]
        .message.content
    )

def read_pdf_by_pypdf(pdf_path: Path) -> list[str]:
    pdf = pypdf.PdfReader(str(pdf_path))
    texts = []
    for page in pdf.pages:
        texts.append(page.extract_text())
    return texts
