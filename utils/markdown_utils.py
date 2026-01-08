import re
import base64
from pathlib import Path

from typing import Any, Dict, List

from base64 import b64decode

class MarkdownMessage:
    """Custom data type to hold OpenAI-compatible message content with embedded images."""
    def __init__(self, content: List[Dict[str, Any]], filename: str = ""):
        self.content = content
        self.filename = filename
        
    def __str__(self):
        """Return a manager-friendly summary (small, no base64)."""
        texts = [item["text"] for item in self.content if item.get("type") == "text"]
        images = [item["image_url"].get("url", "") for item in self.content if item.get("type") == "image_url"]
        # Build short preview of text (first 400 chars)
        preview = " ".join(texts)[:400].replace("\n", " ") + ("…" if sum(len(t) for t in texts) > 400 else "")
        image_list = [f"<image_{idx+1}>" for idx in range(len(images))]
        return (
            f"MarkdownMessage '{self.filename}' | text_preview: '{preview}' | "
            f"images: {', '.join(image_list)} (total {len(images)})"
        )
        
def compressed_image_content(markdownMessage: MarkdownMessage, max_short_side_pixels: int = 1080) -> MarkdownMessage:
    """Return a new MarkdownMessage with images compressed to max_short_side_pixels."""
    from PIL import Image
    import io
    
    new_content = []
    for item in markdownMessage.content:
        if item.get("type") == "image_url":
            url = item["image_url"].get("url", "")
            if url.startswith("data:image"):
                try:
                    base64_part = url.split(",", 1)[1]
                    img_bytes = b64decode(base64_part)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    short_side_pixels = min(img.size)
                    if short_side_pixels >= max_short_side_pixels:
                        scale = max_short_side_pixels / short_side_pixels
                        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                        img = img.resize(new_size, Image.LANCZOS)
                    
                    # Encode back to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"}
                    })
                except Exception as e:
                    new_content.append({
                        "type": "text",
                        "text": f"[Error loading image: {str(e)}]"
                    })
        else:
            new_content.append(item)
    
    return MarkdownMessage(new_content, markdownMessage.filename)
        
        
        
        
def obtain_newContent_and_images_from_markdown(content: str, image_base_dir: str):
    """
    Parse markdown content and extract images, similar to direct_ask.py
    Returns content with image placeholders and a dict of image paths.
    """
    IMG_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    images = {}
    new_content = content
    
    image_counter = 0
    for match in image_pattern.finditer(content):
        image_path = match.group(2)  # Get the path part
        if any(image_path.lower().endswith(ext) for ext in IMG_FILE_EXTENSIONS):
            image_counter += 1
            
            # Resolve relative paths
            if not Path(image_path).is_absolute():
                full_image_path = Path(image_base_dir) / image_path
            else:
                full_image_path = Path(image_path)
                
            if full_image_path.exists():
                current_key = f"<image_{image_counter}>"
                images[current_key] = str(full_image_path)
                new_content = new_content.replace(match.group(0), current_key)
            else:
                # Replace with error message
                new_content = new_content.replace(match.group(0), f"[Image not found: {image_path}]")
    
    return new_content, images
        


def encode_image_to_base64(path: str) -> str:
    """Encode image file to base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
        
        
        
        
        
        


def create_openai_message_content(text: str, image_paths: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Create OpenAI-compatible message content from text with image placeholders.
    Based on the logic from direct_ask.py
    """
    content = []
    
    def append_content(item):
        if content and item["type"] == "text" and content[-1]["type"] == "text":
            content[-1]["text"] += item["text"]
        else:
            content.append(item)
    
    if image_paths:
        pattern = re.compile("|".join(map(re.escape, image_paths.keys())))
        pos = 0
        
        for match in pattern.finditer(text):
            start, end = match.span()
            placeholder = match.group()
            
            # Add text before the image placeholder
            if start > pos:
                append_content({"type": "text", "text": text[pos:start]})
            
            # Add placeholder text so language model sees token
            append_content({"type": "text", "text": f" {placeholder} "})
            # Add the image
            img_path = image_paths[placeholder]
            try:
                # Determine MIME type
                ext = Path(img_path).suffix.lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg', 
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }.get(ext, 'image/jpeg')
                
                encoded = encode_image_to_base64(img_path)
                append_content({
                    "type": "image_url", 
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
                })
            except Exception as e:
                append_content({
                    "type": "text", 
                    "text": f"[Error loading image {img_path}: {str(e)}]"
                })
            
            pos = end
        
        # Add remaining text
        if pos < len(text):
            append_content({"type": "text", "text": text[pos:]})
    else:
        append_content({"type": "text", "text": text})
    
    return content
        
        
        
def load_markdown_from_filepath(file_path: str) -> MarkdownMessage:
        """Parse markdown file and return OpenAI-compatible message content."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            # Use the same logic as direct_ask.py
            image_base_dir = str(file_path.parent)
            new_content, images = obtain_newContent_and_images_from_markdown(
                markdown_text, image_base_dir
            )
            
            # Create OpenAI-compatible message content
            message_content = create_openai_message_content(new_content, images)
            
            return MarkdownMessage(message_content, str(file_path))
            
        except Exception as e:
            # Return error as MarkdownMessage
            return MarkdownMessage([{
                "type": "text",
                "text": f"Error parsing markdown file: {str(e)}"
            }], str(file_path))
            
def load_trace_data_from_filepath(file_path: str = "data_all/non_linear/duffing") -> MarkdownMessage:
    """
    Load trace data from a directory containing images and npz files.
    Returns OpenAI-compatible message content (same as load_markdown_from_filepath).
    
    Args:
        file_path: Path to directory containing sample_X.png and sample_X.npz files.
                   Default is 'data_all/non_linear/duffing'.
    
    Returns:
        MarkdownMessage with all samples combined (text descriptions + embedded images).
    """
    import numpy as np
    import glob
    import re
    
    try:
        dir_path = Path(file_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {file_path}")
        
        # Find all sample IDs by looking for sample_X.png or sample_X.npz files
        all_files = sorted(glob.glob(str(dir_path / "sample_*")))
        # Extract unique sample IDs using regex
        sample_ids = set()
        pattern = re.compile(r'sample_(\d+)\.(png|npz)$')
        for f in all_files:
            match = pattern.search(f)
            if match:
                sample_ids.add(int(match.group(1)))
        
        if not sample_ids:
            return MarkdownMessage([{
                "type": "text",
                "text": f"No sample files found in: {file_path}"
            }], str(dir_path))
        
        # Sort sample IDs
        sample_ids = sorted(sample_ids)
        
        # Collect all text parts and image mappings
        all_text_parts = []
        all_images = {}
        all_npz_data = {}
        
        # Load png and npz files for each sample using a for loop
        for sample_id in sample_ids[:3]:
            # Define file paths for this sample
            png_path = dir_path / f"sample_{sample_id}.png"
            npz_path = dir_path / f"sample_{sample_id}.npz"
            
            # Load npz data and generate description (NpzFile supports dict-like access)
            if npz_path.exists():
                try:
                    with np.load(str(npz_path)) as npz_data:
                        text = _generate_sample_description(sample_id, npz_data)
                except Exception as e:
                    text = _generate_sample_description(sample_id, {"error": str(e)})
            else:
                text = _generate_sample_description(sample_id, {})
            
            # Create image placeholder and add to images dict if png exists
            image_placeholder = f"<image_{sample_id}>"
            if png_path.exists():
                all_images[image_placeholder] = str(png_path)
                all_npz_data[image_placeholder] = str(npz_path)
                all_text_parts.append(f"{text}\n\n{image_placeholder}")
            else:
                all_text_parts.append(text)
        
        # Combine all parts into one content string
        combined_text = "\n\n---\n\n".join(all_text_parts)
        
        # Create OpenAI-compatible message content
        message_content = create_openai_message_content(combined_text, all_images)
        
        return MarkdownMessage(message_content, str(dir_path))
        
    except Exception as e:
        # Return error as MarkdownMessage
        return MarkdownMessage([{
            "type": "text",
            "text": f"Error loading trace data: {str(e)}"
        }], str(file_path))


def _generate_sample_description(sample_id: int, npz_data) -> str:
    """Generate a text description for a trace data sample."""
    import numpy as np
    
    lines = [f"## Sample {sample_id}"]
    
    if not npz_data:
        lines.append("No data available for this sample.")
        return "\n".join(lines)
    
    if "error" in npz_data:
        lines.append(f"Error loading data: {npz_data['error']}")
        return "\n".join(lines)
    
    # Describe each data field
    if "state" in npz_data:
        state = npz_data["state"]
        lines.append(f"- **State data**: shape {state.shape}, range [{state.min():.4f}, {state.max():.4f}]")
    
    # if "mode" in npz_data:
    #     mode = npz_data["mode"]
    #     unique_modes = np.unique(mode)
    #     lines.append(f"- **Mode data**: {len(mode)} time steps, unique modes: {list(unique_modes)}")
    
    if "input" in npz_data:
        input_data = npz_data["input"]
        lines.append(f"- **Input data**: shape {input_data.shape}, range [{input_data.min():.4f}, {input_data.max():.4f}]")
    
    # if "change_points" in npz_data:
    #     cp = npz_data["change_points"]
    #     lines.append(f"- **Change points**: {len(cp)} transitions at indices {list(cp)}")
    
    return "\n".join(lines)
            

def markdown_to_plaintext(markdown_content: MarkdownMessage) -> str:
    """Extract plain text from MarkdownMessage for LLM context (keep image placeholders)."""
    parts: list[str] = []
    for item in markdown_content.content:
        if item.get("type") == "text":
            parts.append(item["text"])
        # image_url blocks are ignored: placeholder already in preceding text
    return "\n".join(parts)
    
def markdown_images_compress(markdown_content: MarkdownMessage, max_short_side_pixels: int = 1000):
    """Extract and compress images from MarkdownMessage, returning list of PIL Images."""
    from PIL import Image
    import io
    images: list[Image.Image] = []  # List to store processed images
    for item in markdown_content.content:
        if item.get("type") == "image_url":
            url = item["image_url"].get("url", "")
            if url.startswith("data:image"):
                try:
                    # Decode base64 image data
                    base64_part = url.split(",", 1)[1]
                    img_bytes = b64decode(base64_part)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # Compress image if it exceeds maximum dimensions
                    short_side_pixels = min(img.size)
                    if short_side_pixels < max_short_side_pixels:
                        img = img  # No compression needed
                    else:
                        scale = max_short_side_pixels / short_side_pixels
                        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                        img = img.resize(new_size, Image.LANCZOS)
                    # Add processed image to list
                    
                    
                    images.append(img)
                except Exception:
                    pass  # Skip images that can't be processed
    return images