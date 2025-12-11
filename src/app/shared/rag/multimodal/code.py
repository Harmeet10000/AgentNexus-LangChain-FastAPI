"""
Converted from Jupyter notebook -> standalone Python script.

System Dependencies (install on your system, not via pip):
    - poppler (poppler-utils)          : PDF processing
    - tesseract (tesseract-ocr)        : OCR for scanned documents
    - libmagic                         : file type detection

Install system deps (examples):
    # for linux
    # sudo apt-get install poppler-utils tesseract-ocr libmagic-dev

    # for mac
    # brew install poppler tesseract libmagic

Python packages used (install in your venv):
    # pip install unstructured[all-docs] langchain_chroma langchain langchain-community langchain-openai python_dotenv
"""

from typing import List, Set
import json
import os

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
# from unstructured.chunking.title import chunk_by_title   # imported in notebook but unused

# LangChain components (imports kept to mirror notebook; adjust/trim as needed)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


def partition_document(file_path: str):
    """
    Extract elements from a PDF using unstructured.partition.pdf.partition_pdf.

    Returns:
        List of elements returned by partition_pdf.
    """
    print(f"📄 Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",  # most accurate extraction (but slower)
        infer_table_structure=True,  # preserve tables as structured HTML
        extract_image_block_types=["Image"],  # capture images found in the PDF
        extract_image_block_to_payload=True,  # store images as base64 payloads
    )

    print(f"✅ Extracted {len(elements)} elements")
    return elements


def element_types(elements: List) -> Set[str]:
    """Return a set of element type strings found in elements."""
    return set([str(type(el)) for el in elements])


def element_to_dict_safe(elements: List, idx: int):
    """
    Safely return elements[idx].to_dict() if available, else return a helpful message.
    """
    if not elements:
        return {"error": "elements list is empty"}
    if idx < 0 or idx >= len(elements):
        return {"error": f"index {idx} out of range (0..{len(elements)-1})"}
    el = elements[idx]
    # many unstructured Elements implement to_dict(); guard around it
    if hasattr(el, "to_dict"):
        try:
            return el.to_dict()
        except Exception as e:
            return {"error": "to_dict() failed", "exception": str(e)}
    # fallback: try to serialize repr
    try:
        return {"repr": repr(el)}
    except Exception as e:
        return {"error": "repr failed", "exception": str(e)}


def main():
    # Change this path to your PDF file
    file_path = "./docs/attention-is-all-you-need.pdf"

    if not os.path.exists(file_path):
        print(f"ERROR: file does not exist: {file_path}")
        return

    elements = partition_document(file_path)

    # show the distinct element types (mirrors notebook output)
    types = element_types(elements)
    print("\nElement types found:")
    print(json.dumps(sorted(list(types)), indent=4))

    # attempt to show one element as dict (notebook showed elements[36].to_dict())
    index_to_inspect = 36
    print(f"\nInspecting element at index {index_to_inspect}:")
    el_dict = element_to_dict_safe(elements, index_to_inspect)
    # pretty-print
    print(json.dumps(el_dict, indent=4, default=str))


if __name__ == "__main__":
    main()

"""
Converted notebook cells -> standalone Python script for:
  - extracting image elements (and saving them)
  - extracting table elements (and saving HTML)
  - creating title-based chunks using unstructured.chunking.title.chunk_by_title

Adjust FILE_PATH at runtime to point to your PDF.
System deps: poppler, tesseract, libmagic (install on your OS as needed).
Python packages: unstructured[all-docs], langchain_chroma, langchain, langchain-openai, python_dotenv
"""

import os
import json
import base64
from typing import List, Dict, Any, Set

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# optional langchain imports retained from notebook (not used directly here)
# from langchain_core.documents import Document
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


def partition_document(file_path: str):
    """Partition a PDF into unstructured elements."""
    print(f"📄 Partitioning document: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    print(f"✅ Extracted {len(elements)} elements")
    return elements


def gather_images(elements: List[Any]) -> List[Any]:
    """Return list of elements whose category is 'Image' (as in notebook)."""
    images = [element for element in elements if getattr(element, "category", None) == "Image"]
    print(f"Found {len(images)} images")
    return images


def _normalize_base64(b64: str) -> bytes:
    """
    Normalize base64 string that might include newlines, data-url prefix, or stray commas.
    Returns decoded bytes. Raises ValueError if decoding fails.
    """
    if not b64:
        raise ValueError("Empty base64 string")
    # Remove data URL prefix if present
    if b64.startswith("data:"):
        # data:<mime>;base64,<payload>
        try:
            b64 = b64.split("base64,", 1)[1]
        except Exception:
            # fallback: strip up to first comma
            b64 = b64.split(",", 1)[-1]
    # Remove whitespace/newlines and stray commas
    b64 = "".join(b64.split())
    b64 = b64.replace(",", "")
    # Add padding if required
    padding = len(b64) % 4
    if padding != 0:
        b64 += "=" * (4 - padding)
    return base64.b64decode(b64)


def save_image_element(image_element: Any, out_dir: str, index: int) -> str:
    """
    Save an image element to out_dir. Returns path to saved file.
    Expects image_element.to_dict() to contain 'metadata' with 'image_base64' and 'image_mime_type'.
    """
    el = image_element
    el_dict = el.to_dict() if hasattr(el, "to_dict") else {}
    meta = el_dict.get("metadata", {}) if isinstance(el_dict, dict) else {}
    b64 = meta.get("image_base64") or meta.get("image_base64_string") or ""
    mime = meta.get("image_mime_type", "application/octet-stream")
    ext = None
    if "/" in mime:
        ext = mime.split("/")[-1].split(";")[0]
    if not ext:
        ext = "bin"
    filename = f"image_{index}.{ext}"
    out_path = os.path.join(out_dir, filename)
    if not b64:
        # if no b64 present, try to fallback to element payload attributes (best-effort)
        raise ValueError("No base64 payload found for image element")
    try:
        data = _normalize_base64(b64)
    except Exception as e:
        raise ValueError(f"Failed to decode image base64: {e}") from e
    with open(out_path, "wb") as fh:
        fh.write(data)
    return out_path


def extract_and_save_images(elements: List[Any], out_dir: str) -> List[str]:
    """Find image elements and save them to out_dir. Returns list of saved file paths."""
    os.makedirs(out_dir, exist_ok=True)
    images = gather_images(elements)
    saved_paths = []
    for idx, img in enumerate(images):
        try:
            path = save_image_element(img, out_dir, idx)
            saved_paths.append(path)
        except Exception as e:
            # continue on error but report
            print(f"Warning: could not save image index {idx}: {e}")
    return saved_paths


def gather_tables(elements: List[Any]) -> List[Any]:
    """Return list of elements whose category is 'Table'."""
    tables = [element for element in elements if getattr(element, "category", None) == "Table"]
    print(f"Found {len(tables)} tables")
    return tables


def save_table_html(table_element: Any, out_dir: str, index: int) -> str:
    """
    Save table element's HTML (if available) to a file.
    Expects table_element.to_dict() to contain metadata['text_as_html'] or .text.
    """
    el_dict = table_element.to_dict() if hasattr(table_element, "to_dict") else {}
    meta = el_dict.get("metadata", {}) if isinstance(el_dict, dict) else {}
    html = meta.get("text_as_html") or el_dict.get("text") or ""
    filename = f"table_{index}.html"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        # If html is absent, write a simple pre block with text fallback
        if html and html.strip().lower().startswith("<table"):
            fh.write(html)
        else:
            fh.write("<pre>\n")
            fh.write(html)
            fh.write("\n</pre>\n")
    return out_path


def extract_and_save_tables(elements: List[Any], out_dir: str) -> List[str]:
    """Find table elements and save HTML/text files. Returns list of saved file paths."""
    os.makedirs(out_dir, exist_ok=True)
    tables = gather_tables(elements)
    saved_paths = []
    for idx, tbl in enumerate(tables):
        try:
            path = save_table_html(tbl, out_dir, idx)
            saved_paths.append(path)
        except Exception as e:
            print(f"Warning: could not save table index {idx}: {e}")
    return saved_paths


def create_chunks_by_title(elements: List[Any], max_characters: int = 3000,
                           new_after_n_chars: int = 2400, combine_text_under_n_chars: int = 500) -> List[Any]:
    """Create intelligent chunks from elements using chunk_by_title (same args as notebook)."""
    print("🔨 Creating smart chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
    )
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def element_type_set(obj_list: List[Any]) -> Set[str]:
    """Return set of string type names for given list of elements or chunks."""
    return set([str(type(x)) for x in obj_list])


def element_to_dict_safe(elements: List[Any], idx: int) -> Dict[str, Any]:
    """Return a safe dict representation (tries to use to_dict) of element at idx."""
    if not elements:
        return {"error": "elements list is empty"}
    if idx < 0 or idx >= len(elements):
        return {"error": f"index {idx} out of range (0..{len(elements)-1})"}
    el = elements[idx]
    if hasattr(el, "to_dict"):
        try:
            return el.to_dict()
        except Exception as e:
            return {"error": "to_dict() failed", "exception": str(e)}
    try:
        return {"repr": repr(el)}
    except Exception as e:
        return {"error": "repr failed", "exception": str(e)}


def main():
    FILE_PATH = "./docs/attention-is-all-you-need.pdf"  # adjust as needed
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: file does not exist: {FILE_PATH}")
        return

    # Partition document
    elements = partition_document(FILE_PATH)

    # -- Images --
    images_out = "./output/images"
    try:
        saved_images = extract_and_save_images(elements, images_out)
        print(f"Saved {len(saved_images)} image files to {images_out}")
    except Exception as e:
        print(f"Error extracting images: {e}")

    # Print a representation of first image element if exists (mirrors notebook images[0].to_dict())
    images = [el for el in elements if getattr(el, "category", None) == "Image"]
    if images:
        try:
            img0 = element_to_dict_safe(images, 0)
            print("\nFirst image element (dict preview):")
            print(json.dumps(img0, indent=4, default=str))
        except Exception as e:
            print(f"Could not print first image dict: {e}")

    # -- Tables --
    tables_out = "./output/tables"
    try:
        saved_tables = extract_and_save_tables(elements, tables_out)
        print(f"Saved {len(saved_tables)} table HTML files to {tables_out}")
    except Exception as e:
        print(f"Error extracting tables: {e}")

    # Show first table dict preview (mirrors notebook tables[0].to_dict())
    tables = [el for el in elements if getattr(el, "category", None) == "Table"]
    if tables:
        try:
            tbl0 = element_to_dict_safe(tables, 0)
            print("\nFirst table element (dict preview):")
            print(json.dumps(tbl0, indent=4, default=str))
        except Exception as e:
            print(f"Could not print first table dict: {e}")

    # -- Create chunks by title --
    try:
        chunks = create_chunks_by_title(elements)
        print("\nUnique chunk types:")
        print(json.dumps(sorted(list(element_type_set(chunks))), indent=4))
    except Exception as e:
        print(f"Error creating chunks: {e}")
        chunks = []

    # Show a sample chunk (first) as dict if possible
    if chunks:
        try:
            chunk0 = element_to_dict_safe(chunks, 0)
            print("\nFirst chunk (dict preview):")
            print(json.dumps(chunk0, indent=4, default=str))
        except Exception as e:
            print(f"Could not print chunk preview: {e}")


if __name__ == "__main__":
    main()
"""
Replicates the notebook cell that inspects:
    chunks[11].metadata.orig_elements[-1].to_dict()

This script assumes:
    - You already created `elements` from partition_pdf()
    - You already created `chunks` using chunk_by_title()

If not, integrate this into the main script I wrote earlier.
"""

import json
from typing import Any, Dict, List


def safe_to_dict(obj: Any) -> Dict:
    """Safely convert an element or chunk to a dict."""
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception as e:
            return {"error": "to_dict() failed", "exception": str(e)}

    # Fallback if to_dict() does not exist
    try:
        return {"repr": repr(obj)}
    except Exception as e:
        return {"error": "repr failed", "exception": str(e)}


def inspect_original_element_from_chunk(chunks: List[Any], chunk_index: int, elem_index: int):
    """
    Replicates the notebook behavior:

        chunks[11].metadata.orig_elements[-1].to_dict()

    With safety checks.
    """
    if not chunks:
        print("ERROR: chunks list is empty")
        return

    if chunk_index >= len(chunks) or chunk_index < 0:
        print(f"ERROR: chunk index {chunk_index} out of range (0..{len(chunks)-1})")
        return

    chunk = chunks[chunk_index]

    if not hasattr(chunk, "metadata"):
        print("ERROR: chunk has no metadata attribute")
        return

    meta = getattr(chunk, "metadata")
    if not hasattr(meta, "orig_elements"):
        print("ERROR: chunk.metadata has no orig_elements attribute")
        return

    orig_els = meta.orig_elements
    if not orig_els:
        print("ERROR: chunk.metadata.orig_elements is empty")
        return

    # Support negative indexing (-1)
    if elem_index >= len(orig_els) or elem_index < -len(orig_els):
        print(f"ERROR: element index {elem_index} out of range for orig_elements list")
        return

    target = orig_els[elem_index]
    result = safe_to_dict(target)

    # Pretty print
    print(json.dumps(result, indent=4, default=str))


def main():
    """
    IMPORTANT:
        You MUST have created `elements` and `chunks` already.
        This function expects that `chunks` exists in the surrounding context.

    Typically:
        elements = partition_document(FILE_PATH)
        chunks = create_chunks_by_title(elements)
    """

    # You should import or generate `chunks` before running this.
    # Here we assume they exist in the imported context.
    try:
        from your_previous_script_namespace import chunks  # adjust to your actual script/module
    except Exception:
        print("ERROR: You must load `chunks` before calling inspect function.")
        return

    # Notebook equivalent:
    inspect_original_element_from_chunk(chunks, chunk_index=11, elem_index=-1)


if __name__ == "__main__":
    main()
