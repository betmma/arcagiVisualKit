# generate_video_output(input_image_path, prompt_text) -> output_directory_path
# generate_video_output_multiple_tries(input_image_path, prompt_text, attempts=3) -> output_directory_path
# generate_video_outputs_multiprocess(image_paths_list, prompt_texts, processes=None, attempts=1, chunksize=1) -> list of output_directory_path
from __future__ import annotations

import base64, json, multiprocessing as _mp, os, re, time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any

import cv2, requests


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
#
# The public generation functions use these defaults:
#
#   generate_video_output(...)
#   generate_video_output_multiple_tries(...)
#   generate_video_outputs_multiprocess(...)
#
# API key:
#   Option 1: put the key in the first line of API_KEY_PATH.
#   Option 2: set API_KEY directly to the key string.
#
# Model / base URL:
#   Edit MODEL_NAME and BASE_URL below.
#
API_KEY_PATH = "api_key.txt"
API_KEY = ""
MODEL_NAME = "omni-fast"#"veo_3_1-landscape"#
BASE_URL = "https://jyapi.ai-wx.cn/v1"
OUTPUT_ROOT = os.path.join("evaluation", "output")

DEFAULT_INPUT_IMAGE = r"data/mirror/puzzles/c92274c1-deae-4f22-ab1a-ae5a8039694f_puzzle.png"
DEFAULT_PROMPT = "Instantly reflect this pattern along the central, vertical axis while keeping the existing colored pattern without modification. Static camera perspective, no zoom or pan."

IMAGE_DATA_RE = re.compile(r"data:image/([^;]+);base64,([A-Za-z0-9+/=]+)")
URL_RE = re.compile(r"https?://[^\s<>\")\]`]+", re.IGNORECASE)
IMAGE_URL_RE = re.compile(r"https?://[^\s<>\")\]`]+?\.(?:png|jpg|jpeg|gif)(?:\?[^\s<>\")\]`]*)?", re.IGNORECASE)
VIDEO_EXTENSIONS = ("mp4", "webm", "mov", "mkv")
VIDEO_CONTENT_SUFFIXES = ("/content", "/download")


@dataclass
class VideoApiConfig:
    """Configuration for the video generation API."""
    api_key_path: str = API_KEY_PATH
    api_key: str = API_KEY
    model: str = MODEL_NAME
    base_url: str = BASE_URL
    output_root: str = OUTPUT_ROOT
    timeout: int = 600
    use_stream: bool = True
    request_retries: int = 1
    retry_delay: float = 0.0
    duration: int = 15
    no_proxy: bool = True
    debug: bool = False


def read_api_key(path: str) -> str:
    """Read the first line of an API key file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()


def load_config(config: VideoApiConfig | None = None) -> VideoApiConfig:
    """Return a usable config without doing work at import time."""
    config = replace(config) if config is not None else VideoApiConfig()
    if not config.api_key:
        config.api_key = read_api_key(config.api_key_path)
    return config


def public_config(config: VideoApiConfig) -> dict[str, Any]:
    """Return config data safe to write into metadata."""
    data = asdict(config)
    data["api_key"] = "***" if data["api_key"] else ""
    return data


def normalize_image_paths(input_image_path) -> list[str]:
    """Normalize one image path or a sequence of image paths into a list."""
    if isinstance(input_image_path, (list, tuple)):
        image_paths = [str(path) for path in input_image_path if path]
    elif isinstance(input_image_path, str):
        image_paths = [input_image_path]
    else:
        raise TypeError("input_image_path must be a string or a sequence of strings")
    if not image_paths:
        raise ValueError("No input image paths provided")
    return image_paths


def image_mime_type(image_path: str) -> str:
    """Return the MIME type to use for a local image."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def image_to_data_url(image_path: str) -> str:
    """Read a local image and return an image data URL."""
    with open(image_path, "rb") as img_file:
        encoded_data = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{image_mime_type(image_path)};base64,{encoded_data}"


def build_messages(image_paths: list[str], prompt_text: str) -> list[dict[str, Any]]:
    """Build OpenAI-style chat messages containing text and image inputs."""
    image_contents = []
    for image_path in image_paths:
        image_contents.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    return [{"role": "user", "content": [{"type": "text", "text": prompt_text}, *image_contents]}]


def create_output_directory(output_root: str) -> str:
    """Create and return a unique output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    os.makedirs(output_root, exist_ok=True)
    candidate = os.path.join(output_root, f"output_{timestamp}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_root, f"output_{timestamp}_{counter:02d}")
        counter += 1
    os.makedirs(candidate, exist_ok=False)
    return candidate


def write_json(path: str, data: Any) -> None:
    """Write JSON to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    """Write text to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def request_with_retries(method: str, url: str, *, attempts: int, retry_delay: float, **kwargs) -> requests.Response:
    """Run an HTTP request with simple retry handling."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method, url, proxies={}, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt == attempts:
                break
            if retry_delay > 0:
                time.sleep(retry_delay)
    raise last_error


def call_api_raw(config: VideoApiConfig, messages: list[dict[str, Any]], output_dir: str) -> dict[str, Any]:
    """Call the chat-completions endpoint through raw HTTP."""
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {"model": config.model, "messages": messages, "stream": config.use_stream, "duration": config.duration}
    response = request_with_retries("POST", url, attempts=config.request_retries, retry_delay=config.retry_delay, headers=headers, json=payload, timeout=config.timeout, stream=config.use_stream)
    if config.use_stream:
        json_response = read_streaming_response(response, output_dir)
    else:
        json_response = response.json()
    write_json(os.path.join(output_dir, "input_messages.json"), messages)
    write_json(os.path.join(output_dir, "raw_api_response.json"), json_response)
    return json_response


def read_streaming_response(response: requests.Response, output_dir: str) -> dict[str, Any]:
    """Read a streaming OpenAI-compatible response into a normal response dict."""
    full_content = ""
    chunks = []
    for line in response.iter_lines():
        if not line:
            continue
        line_text = line.decode("utf-8")
        if not line_text.startswith("data: "):
            continue
        data_text = line_text[6:]
        if data_text == "[DONE]":
            continue
        try:
            chunk = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        chunks.append(chunk)
        choices = chunk.get("choices", [])
        if choices:
            full_content += choices[0].get("delta", {}).get("content", "")
    write_json(os.path.join(output_dir, "stream_chunks.json"), chunks)
    return {"choices": [{"message": {"role": "assistant", "content": full_content}}], "stream_chunks": chunks}


def call_openai_client(config: VideoApiConfig, messages: list[dict[str, Any]]) -> Any:
    """Call the API through the OpenAI client as a fallback."""
    from openai import OpenAI
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    last_error = None
    for attempt in range(1, config.request_retries + 1):
        try:
            return client.chat.completions.create(model=config.model, messages=messages, timeout=config.timeout)
        except Exception as error:
            last_error = error
            if attempt == config.request_retries:
                break
            if config.retry_delay > 0:
                time.sleep(config.retry_delay)
    raise last_error


def call_api_with_fallback(config: VideoApiConfig, messages: list[dict[str, Any]], output_dir: str) -> Any:
    """Call raw HTTP first, then fall back to the OpenAI client."""
    try:
        return call_api_raw(config, messages, output_dir)
    except Exception as error:
        write_text(os.path.join(output_dir, "raw_api_error.txt"), str(error))
        response = call_openai_client(config, messages)
        write_text(os.path.join(output_dir, "openai_client_response.txt"), repr(response))
        return response


def append_message_media_fields(content: str, message: dict[str, Any]) -> str:
    """Append likely media fields from a message dict into the parseable content string."""
    for field_name in ("images", "image", "attachments", "media", "files", "data"):
        value = message.get(field_name)
        if not value:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str):
                content += "\n" + (item if item.startswith("data:") or item.startswith("http") else f"data:image/png;base64,{item}")
            elif isinstance(item, dict):
                if item.get("url"):
                    content += "\n" + item["url"]
                elif item.get("data"):
                    content += f"\ndata:image/png;base64,{item['data']}"
                elif item.get("base64"):
                    content += f"\ndata:image/png;base64,{item['base64']}"
    return content


def extract_response_content(response: Any) -> str:
    """Extract text plus likely media payloads from a raw dict or OpenAI response object."""
    if isinstance(response, dict):
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or ""
        return append_message_media_fields(content, message)
    message = response.choices[0].message
    content = message.content or ""
    images = getattr(message, "images", None)
    if images:
        for image in images:
            if isinstance(image, str):
                content += f"\ndata:image/png;base64,{image}"
            elif hasattr(image, "data"):
                content += f"\ndata:image/png;base64,{image.data}"
    return content


def unique_matches(pattern: re.Pattern, text: str) -> list[str]:
    """Return unique regex matches in first-seen order."""
    values = []
    for match in pattern.finditer(text):
        value = match.group(0)
        if value not in values:
            values.append(value)
    return values


def clean_url(url: str) -> str:
    """Remove punctuation commonly attached to URLs in prose."""
    return url.rstrip(".,;:!?")


def unique_urls(text: str) -> list[str]:
    """Return unique HTTP URLs in first-seen order."""
    values = []
    for match in URL_RE.finditer(text):
        value = clean_url(match.group(0))
        if value not in values:
            values.append(value)
    return values


def is_likely_video_url(url: str) -> bool:
    """Return whether a URL points at a video file or video download endpoint."""
    url_without_query = url.split("?", 1)[0].rstrip("/")
    url_path_lower = url_without_query.lower()
    ext = os.path.splitext(url_path_lower)[1].lstrip(".")
    if ext in VIDEO_EXTENSIONS:
        return True
    if "/videos/" in url_path_lower or "/video/" in url_path_lower:
        return url_path_lower.endswith(VIDEO_CONTENT_SUFFIXES) or "download" in url.lower()
    return False


def extract_video_urls(content: str) -> list[str]:
    """Extract downloadable video URLs, including extensionless content endpoints."""
    return [url for url in unique_urls(content) if is_likely_video_url(url)]


def save_base64_image(data_url: str, output_dir: str, image_index: int) -> str:
    """Save one base64 image data URL."""
    match = IMAGE_DATA_RE.match(data_url)
    if match is None:
        raise ValueError("invalid image data URL")
    media_subtype = match.group(1).lower()
    ext = "jpg" if media_subtype in ("jpeg", "jpg") else media_subtype
    image_data = base64.b64decode(match.group(2))
    image_path = os.path.join(output_dir, f"image_{image_index}.{ext}")
    with open(image_path, "wb") as img_file:
        img_file.write(image_data)
    return image_path


def extension_from_response(url: str, response: requests.Response, default_ext: str) -> str:
    """Choose a file extension from URL and content type."""
    url_ext = os.path.splitext(url.split("?", 1)[0])[1].lstrip(".").lower()
    if url_ext:
        return url_ext
    content_type = response.headers.get("content-type", "").lower()
    for ext in ("mp4", "webm", "mov", "mkv", "png", "jpg", "jpeg", "gif"):
        if ext in content_type:
            return "jpg" if ext == "jpeg" else ext
    return default_ext


def download_url(url: str, output_dir: str, filename_prefix: str, index: int, default_ext: str, timeout: int) -> str:
    """Download a URL to the output directory."""
    response = request_with_retries("GET", url, attempts=1, retry_delay=0, stream=True, timeout=timeout)
    ext = extension_from_response(url, response, default_ext)
    path = os.path.join(output_dir, f"{filename_prefix}_{index}.{ext}")
    with open(path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)
    return path


def extract_last_frame(video_path: str, output_dir: str) -> str | None:
    """Extract the last frame of a video as result.png."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    result_path = os.path.join(output_dir, "result.png")
    return result_path if cv2.imwrite(result_path, frame) else None


def save_response_artifacts(content: str, output_dir: str, config: VideoApiConfig) -> dict[str, Any]:
    """Save response text, media downloads, video downloads, and final frame artifacts."""
    artifacts: dict[str, Any] = {"content_path": os.path.join(output_dir, "content.txt"), "original_content_path": os.path.join(output_dir, "original_content.txt"), "images": [], "videos": [], "result_frame_path": None}
    text_content = content
    for image_index, match in enumerate(list(IMAGE_DATA_RE.finditer(content)), start=1):
        data_url = match.group(0)
        saved_path = save_base64_image(data_url, output_dir, image_index)
        artifacts["images"].append(saved_path)
        text_content = text_content.replace(data_url, f"[saved image: {saved_path}]")
    next_image_index = len(artifacts["images"]) + 1
    for image_url in unique_matches(IMAGE_URL_RE, content):
        saved_path = download_url(image_url, output_dir, "image_url", next_image_index, "png", config.timeout)
        artifacts["images"].append(saved_path)
        text_content = text_content.replace(image_url, f"[downloaded image: {saved_path}]")
        next_image_index += 1
    video_urls = extract_video_urls(content)
    video_urls.sort(key=lambda url: 0 if "download" in url.lower() else 1)
    for video_index, video_url in enumerate(video_urls, start=1):
        saved_path = download_url(video_url, output_dir, "video", video_index, "mp4", config.timeout)
        artifacts["videos"].append(saved_path)
        text_content = text_content.replace(video_url, f"[downloaded video: {saved_path}]")
        if artifacts["result_frame_path"] is None:
            artifacts["result_frame_path"] = extract_last_frame(saved_path, output_dir)
    if artifacts["result_frame_path"]:
        text_content += f"\nExtracted last frame: {artifacts['result_frame_path']}\n"
    write_text(artifacts["content_path"], text_content)
    write_text(artifacts["original_content_path"], content)
    return artifacts


def write_metadata(output_dir: str, config: VideoApiConfig, image_paths: list[str], prompt_text: str, artifacts: dict[str, Any] | None, success: bool, error: str | None = None) -> None:
    """Write metadata for one generation run."""
    metadata = {"success": success, "error": error, "created_at": datetime.now().isoformat(), "input_images": image_paths, "prompt": prompt_text, "config": public_config(config), "artifacts": artifacts or {}}
    write_json(os.path.join(output_dir, "metadata.json"), metadata)


def generate_video_output(input_image_path, prompt_text):
    """Generate one video output and return the output directory path."""
    config = load_config()
    if config.no_proxy:
        os.environ["NO_PROXY"] = "*"
    image_paths = normalize_image_paths(input_image_path)
    output_dir = create_output_directory(config.output_root)
    artifacts = None
    try:
        messages = build_messages(image_paths, prompt_text)
        response = call_api_with_fallback(config, messages, output_dir)
        content = extract_response_content(response)
        write_text(os.path.join(output_dir, "response_content_preview.txt"), content[:4000])
        artifacts = save_response_artifacts(content, output_dir, config)
        write_metadata(output_dir, config, image_paths, prompt_text, artifacts, True)
        return output_dir
    except Exception as error:
        write_metadata(output_dir, config, image_paths, prompt_text, artifacts, False, str(error))
        raise


def generate_video_output_multiple_tries(input_image_path, prompt_text, attempts=3):
    """Try whole video generations until result.png exists or attempts are exhausted."""
    result = None
    for attempt in range(1, attempts + 1):
        try:
            result = generate_video_output(input_image_path, prompt_text)
            result_png = os.path.join(result, "result.png")
            if not os.path.exists(result_png):
                raise FileNotFoundError(f"Expected result frame not found at {result_png}")
            return result
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2)
    return result


def _mp_worker_generate(arg_tuple):
    """Generate one output inside a multiprocessing worker."""
    input_image_path, prompt_text, attempts = arg_tuple
    if attempts and attempts > 1:
        return generate_video_output_multiple_tries(input_image_path, prompt_text, attempts=attempts)
    return generate_video_output(input_image_path, prompt_text)


def generate_video_outputs_multiprocess(image_paths_list, prompt_texts, processes=None, attempts=1, chunksize=1):
    """Generate multiple video outputs in parallel using multiprocessing."""
    if not isinstance(image_paths_list, (list, tuple)) or not isinstance(prompt_texts, (list, tuple)):
        raise TypeError("image_paths_list and prompt_texts must be lists/tuples")
    if len(image_paths_list) != len(prompt_texts):
        raise ValueError("image_paths_list and prompt_texts must have the same length")
    if attempts is None or attempts < 1:
        attempts = 1
    tasks = [(image_paths_list[i], prompt_texts[i], attempts) for i in range(len(prompt_texts))]
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=processes) as pool:
        return pool.map(_mp_worker_generate, tasks, chunksize=chunksize)


def main() -> None:
    """Run the default example generation."""
    output_dir = generate_video_output(DEFAULT_INPUT_IMAGE, DEFAULT_PROMPT)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
