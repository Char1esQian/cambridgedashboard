#!/usr/bin/env python3
"""
Automated cafe menu updater.

Workflow:
1) Weekly: fetch the source menu once and extract structured JSON with Gemini.
2) Daily: reuse existing menu.json and only generate/reuse today's needed image.
3) Save generated images under assets/menu-generated/ and reusable archives under assets/menu-archive/.
4) Write image URLs back into menu.json only when meaningful content changes.
"""

import argparse
import ast
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw

MENU_SOURCE_URL = os.environ.get("MENU_SOURCE_URL", "https://cafe.sebastians.com/sebclients/3130alewife.jpg")
REQUIRED_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CATEGORY_PRIORITY = ["Carving", "Plant Power", "Action"]
DEFAULT_CANVAS = (1280, 720)
VOLATILE_METADATA_KEYS = {"updatedAt", "sourceFetchedAt"}

# Gemini models. Defaults are chosen for lowest practical cost.
# Override in GitHub Actions if needed, e.g. GEMINI_EXTRACTION_MODEL=gemini-2.5-flash.
EXTRACTION_MODEL = os.environ.get("GEMINI_EXTRACTION_MODEL", "gemini-2.5-flash-lite")
EXTRACTION_MODEL_FALLBACK = os.environ.get("GEMINI_EXTRACTION_MODEL_FALLBACK", "gemini-2.5-flash")
EXTRACTION_MODEL_CANDIDATES = [
    model for model in [EXTRACTION_MODEL, EXTRACTION_MODEL_FALLBACK] if model
]

# Cheapest current Google image-generation default.
# Set GEMINI_IMAGE_MODEL_FALLBACK if you want a more expensive backup model.
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
IMAGE_MODEL_FALLBACK = os.environ.get("GEMINI_IMAGE_MODEL_FALLBACK", "")
IMAGE_GEN_MAX_RETRIES = 4
MENU_TIMEZONE = "America/New_York"
IMAGE_MODEL_CANDIDATES = [
    model for model in [IMAGE_MODEL, IMAGE_MODEL_FALLBACK] if model
]

# Optional: set GEMINI_SERVICE_TIER=flex in GitHub Actions for lower cost with best-effort availability.
# Leave unset for standard synchronous API behavior.
GEMINI_SERVICE_TIER = os.environ.get("GEMINI_SERVICE_TIER", "standard")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MENU_PATH = PROJECT_ROOT / "menu.json"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets" / "menu-generated"
DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "assets" / "menu-archive"
TUESDAY_REFERENCE_IMAGE = PROJECT_ROOT / "assets" / "menu-generated" / "taco-tuesday-reference.png"
CAULIFLOWER_FLATBREAD_ARCHIVE_NAME = "charred-cauliflower-flatbread.png"

EXTRACTION_PROMPT = """Analyze this cafe menu source and extract ALL menu items into a structured JSON format.

The menu should be organized by day of the week (Monday through Friday).
For each day, extract the available menu categories and their items.

Common categories include:
- Breakfast (morning items like eggs, sandwiches)
- Soup (soup of the day)
- Deli (sandwiches, wraps)
- Carving (main entrees, carved meats)
- Charred (flatbreads, pizzas)
- Plant Power (vegetarian/vegan options)
- Action (special stations, build-your-own)

For each menu item, provide:
- name: The dish name
- description: Additional details, ingredients, or sides (empty string if none)
- price: The price(s) listed (use format like "$8.95" or "$2.90-$4.95" for ranges)

Return ONLY valid JSON in this exact format, no markdown code blocks.
"""

FOOD_PHOTO_PROMPT_TEMPLATE = """Photoreal food product photo: a single black rectangular meal-prep container with a clear lid slightly open, on a clean off-white table. Modern cafeteria background, heavily blurred chairs, soft daylight from left, shallow depth of field, minimal/neutral colors, lots of negative space, slightly top-down view (around 35 to 45 degrees), crisp focus on food so ingredients are clearly visible.

Meal based on: {item_name}. Proteins: {proteins_list}. Ingredients: {ingredients_list}. Base/crust: {base_or_crust}. Sauce: {sauce_or_none}. {protein_layout_instruction} Make it realistic, neatly arranged inside the tray, appetizing, no props.

Negative: text, logos, labels, watermarks, hands/people, utensils, clutter, cartoon/anime, oversaturated, distorted container/lid.
"""

TUESDAY_TACO_SPECIAL = {
    "category": "Taco Tuesday",
    "item": {
        "name": "Taco Tuesday Bowl",
        "description": "Seasoned chicken and beef, shredded lettuce, diced tomato, red onion, cheddar, salsa, sour cream, soft tortillas",
        "price": "Market Price",
    },
}


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized or "item"


def normalize_price(value: str) -> str:
    return str(value or "").replace("\u2013", "-").replace("â€“", "-").strip()


def parse_stringified_dict(value):
    text = str(value or "").strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def get_extraction_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or ""


def get_image_generation_api_key() -> str:
    return (
        os.environ.get("GEMINI_API_KEY_NANO")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    )


def build_generate_config(**kwargs) -> dict:
    """Build Gemini generate_content config, optionally adding Flex/standard/priority tier."""
    config = {key: value for key, value in kwargs.items() if value is not None}
    if GEMINI_SERVICE_TIER in {"flex", "standard", "priority"}:
        config["service_tier"] = GEMINI_SERVICE_TIER
    return config


def guess_mime_type(url: str, response: requests.Response) -> str:
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type:
        return content_type
    guessed, _ = mimetypes.guess_type(url)
    return guessed or "application/octet-stream"


def fetch_menu_source() -> tuple[bytes, str, str]:
    print(f"Fetching menu source from {MENU_SOURCE_URL}...", file=sys.stderr)
    response = requests.get(MENU_SOURCE_URL, timeout=30)
    response.raise_for_status()
    source_bytes = response.content
    mime_type = guess_mime_type(MENU_SOURCE_URL, response)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if mime_type.startswith("image/"):
        image = Image.open(BytesIO(source_bytes))
        print(f"Image downloaded: {image.size[0]}x{image.size[1]}", file=sys.stderr)
    else:
        print(f"Source downloaded: {len(source_bytes)} bytes ({mime_type})", file=sys.stderr)
    return source_bytes, mime_type, source_hash


def extract_menu_with_gemini(source_bytes: bytes, mime_type: str) -> tuple[str, str]:
    from google import genai
    from google.genai import types

    api_key = get_extraction_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    print("Sending menu source to Gemini extraction API...", file=sys.stderr)
    client = genai.Client(api_key=api_key)
    source_part = types.Part.from_bytes(data=source_bytes, mime_type=mime_type)

    last_error = None
    for model_name in EXTRACTION_MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[EXTRACTION_PROMPT, source_part],
                config=build_generate_config(response_mime_type="application/json"),
            )
            print(f"Extracted menu JSON with model: {model_name}", file=sys.stderr)
            raw_text = (response.text or "").strip()
            if raw_text.startswith("```"):
                lines = [line for line in raw_text.split("\n") if not line.startswith("```")]
                raw_text = "\n".join(lines)
            return raw_text, model_name
        except Exception as err:
            last_error = err
            err_text = str(err)
            if is_model_not_found_error(err_text):
                print(f"Extraction model unavailable: {model_name}", file=sys.stderr)
                continue
            raise

    raise RuntimeError(f"Menu extraction failed: {last_error}")


def validate_menu_json(menu_json: dict) -> dict:
    def coerce_menu_item(item):
        if isinstance(item, dict):
            nested = None
            if isinstance(item.get("name"), dict):
                nested = item.get("name")
            elif isinstance(item.get("name"), str):
                nested = parse_stringified_dict(item.get("name"))
            source = nested if isinstance(nested, dict) else item

            name = source.get("name") or source.get("title") or source.get("item") or source.get("dish")
            if not name:
                return None
            normalized = {
                "name": str(name).strip(),
                "description": str(source.get("description") or source.get("details") or "").strip(),
                "price": normalize_price(source.get("price") or source.get("cost") or "Market Price"),
            }
            image_url = source.get("imageUrl") or item.get("imageUrl") or item.get("image")
            if image_url:
                normalized["imageUrl"] = str(image_url)
            return normalized
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            parsed = parse_stringified_dict(text)
            if isinstance(parsed, dict):
                return coerce_menu_item(parsed)
            return {"name": text, "description": "", "price": "Market Price"}
        if isinstance(item, list):
            parts = [str(part).strip() for part in item if str(part).strip()]
            if not parts:
                return None
            return {"name": ", ".join(parts), "description": "", "price": "Market Price"}
        return None

    for day in REQUIRED_DAYS:
        if day not in menu_json:
            print(f"Warning: Missing day '{day}' in menu", file=sys.stderr)
            continue
        day_menu = menu_json[day]
        if not isinstance(day_menu, dict):
            raise ValueError(f"Invalid menu structure for {day}")
        normalized_day_menu = {}
        for category, item in day_menu.items():
            normalized_item = coerce_menu_item(item)
            if not normalized_item:
                print(f"Warning: Skipping invalid item structure for {day}/{category}", file=sys.stderr)
                continue
            normalized_day_menu[category] = normalized_item
        menu_json[day] = normalized_day_menu
    return menu_json


def find_category_entry(day_menu: dict, target_category: str):
    target = target_category.lower()
    for category, item in day_menu.items():
        if str(category).lower() == target and isinstance(item, dict):
            return category, item
    return None, None


def select_daily_highlights(day_menu: dict):
    highlights = []
    for category in CATEGORY_PRIORITY:
        matched_category, item = find_category_entry(day_menu, category)
        if matched_category and item:
            highlights.append((matched_category, item))
    if highlights:
        return highlights
    first = next(((c, i) for c, i in day_menu.items() if isinstance(i, dict)), (None, None))
    return [first] if first[0] and first[1] else []


def render_fallback_tray_image(output_path: Path):
    width, height = DEFAULT_CANVAS
    base = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(base)
    for y in range(height):
        shade = int(245 - ((y / max(height - 1, 1)) * 16))
        draw.line([(0, y), (width, y)], fill=(shade, shade, shade))
    draw.rounded_rectangle((330, 250, 950, 560), radius=34, fill=(24, 24, 24))
    draw.rounded_rectangle((370, 290, 910, 520), radius=24, fill=(248, 248, 248))
    draw.rounded_rectangle((390, 310, 890, 500), radius=18, outline=(220, 220, 220), width=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(output_path, format="PNG", optimize=True)


def render_cauliflower_flatbread_stock_image(output_path: Path):
    width, height = DEFAULT_CANVAS
    base = Image.new("RGB", (width, height), (238, 235, 226))
    draw = ImageDraw.Draw(base)

    for y in range(height):
        shade = int(248 - ((y / max(height - 1, 1)) * 28))
        draw.line([(0, y), (width, y)], fill=(shade, shade - 2, shade - 7))

    # Soft cafeteria shapes in the background, intentionally abstract and token-free.
    for x, y, w, h, fill in [
        (70, 70, 210, 80, (214, 205, 190)),
        (350, 48, 260, 92, (225, 217, 204)),
        (780, 62, 280, 88, (208, 202, 190)),
        (145, 170, 1020, 18, (198, 188, 174)),
    ]:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill)

    tray = (250, 240, 1030, 610)
    draw.rounded_rectangle(tray, radius=46, fill=(23, 23, 22))
    draw.rounded_rectangle((288, 276, 992, 574), radius=34, fill=(244, 244, 238))

    crust = (345, 315, 935, 540)
    draw.rounded_rectangle(crust, radius=58, fill=(225, 198, 143), outline=(194, 151, 88), width=5)
    draw.rounded_rectangle((365, 334, 915, 522), radius=48, fill=(236, 218, 169))

    # Pesto, char, cauliflower, mushroom, and onion details.
    for x, y in [(418, 370), (500, 424), (585, 360), (710, 425), (810, 372), (872, 470)]:
        draw.ellipse((x, y, x + 45, y + 28), fill=(74, 128, 67))
    for x, y in [(454, 460), (635, 476), (760, 338), (866, 418)]:
        draw.ellipse((x, y, x + 38, y + 18), fill=(83, 55, 34))
    for x, y in [(430, 410), (545, 382), (660, 414), (745, 472), (835, 444)]:
        draw.ellipse((x, y, x + 62, y + 52), fill=(235, 232, 211), outline=(181, 174, 142), width=3)
        draw.arc((x + 10, y + 8, x + 48, y + 42), start=210, end=35, fill=(166, 158, 127), width=2)
    for x, y in [(520, 472), (604, 350), (706, 392), (815, 494)]:
        draw.arc((x, y, x + 70, y + 34), start=180, end=350, fill=(124, 67, 112), width=5)

    lid = (318, 245, 966, 420)
    draw.rounded_rectangle(lid, radius=32, outline=(222, 225, 224), width=8)
    draw.line((342, 258, 942, 407), fill=(255, 255, 255), width=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(output_path, format="PNG", optimize=True)


def get_archive_dir_for_assets(assets_dir: Path) -> Path:
    return assets_dir.parent / DEFAULT_ARCHIVE_DIR.name


def get_menu_image_archive_key(item: dict) -> str:
    name = str(item.get("name") or "").strip()
    return slugify(name)[:96]


def get_legacy_menu_image_archive_key(item: dict) -> str:
    legacy_name = str({
        "name": str(item.get("name") or "").strip(),
        "description": str(item.get("description") or "").strip(),
        "price": str(item.get("price") or "").strip(),
    })
    return slugify(legacy_name)[:96]


def is_cauliflower_flatbread_item(category: str, item: dict) -> bool:
    text = " ".join([
        str(category or ""),
        str(item.get("name") or ""),
        str(item.get("description") or ""),
    ]).lower()
    if "cauliflower" not in text:
        return False
    return any(marker in text for marker in ("flatbread", "crust", "pizza", "charred", "plant power"))


def get_reusable_image_paths(category: str, item: dict, assets_dir: Path) -> list[Path]:
    archive_dir = get_archive_dir_for_assets(assets_dir)
    if is_cauliflower_flatbread_item(category, item):
        path = archive_dir / CAULIFLOWER_FLATBREAD_ARCHIVE_NAME
        if not path.exists():
            render_cauliflower_flatbread_stock_image(path)
            print(f"Created reusable cauliflower flatbread stock image: {to_repo_relative_url(path)}", file=sys.stderr)
        return [path]

    keys = [get_menu_image_archive_key(item), get_legacy_menu_image_archive_key(item)]
    unique_keys = []
    for key in keys:
        if key and key not in unique_keys:
            unique_keys.append(key)
    return [archive_dir / f"{key}.png" for key in unique_keys]


def copy_reusable_image(source_path: Path, output_path: Path) -> bool:
    if not source_path.exists():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output_path)
    print(f"Reused archived menu image: {to_repo_relative_url(source_path)}", file=sys.stderr)
    return True


def split_meal_components(item: dict) -> dict:
    name = str(item.get("name", "")).strip()
    description = str(item.get("description", "")).strip()
    raw = ", ".join(part for part in [name, description] if part)
    tokens = [t.strip() for t in raw.split(",") if t.strip()]

    base_keywords = ("rice", "noodle", "pasta", "crust", "flatbread", "quinoa", "potato")
    sauce_keywords = ("sauce", "aioli", "pesto", "gravy", "chimichurri", "marinara", "curry", "dressing")
    protein_keywords = (
        "chicken",
        "beef",
        "pork",
        "salmon",
        "tuna",
        "shrimp",
        "turkey",
        "tofu",
        "lamb",
        "sausage",
        "meatball",
        "ham",
        "cod",
        "tilapia",
        "fish",
        "steak",
        "chorizo",
        "egg",
    )

    base = "none"
    sauce = "none"
    ingredients = []
    proteins = []

    for token in tokens:
        lower = token.lower()
        if base == "none" and any(k in lower for k in base_keywords):
            base = token
            continue
        if sauce == "none" and any(k in lower for k in sauce_keywords):
            sauce = token
            continue
        if any(k in lower for k in protein_keywords):
            if token not in proteins:
                proteins.append(token)
            continue
        ingredients.append(token)

    if not ingredients and name:
        ingredients = [name]

    proteins_list = ", ".join(proteins) if proteins else "unspecified"
    if len(proteins) >= 2:
        protein_layout_instruction = (
            "If multiple proteins are listed, place each protein in a separate smaller side container "
            "positioned next to the main dish container, and do not mix the proteins together."
        )
    elif len(proteins) == 1:
        protein_layout_instruction = "Use the listed protein as the primary protein in the main dish."
    else:
        protein_layout_instruction = (
            "No explicit protein is listed; keep plating realistic to the named dish without adding extra props."
        )

    return {
        "item_name": name or "Chef special",
        "ingredients_list": ", ".join(ingredients) if ingredients else "Chef selected seasonal ingredients",
        "proteins_list": proteins_list,
        "base_or_crust": base,
        "sauce_or_none": sauce,
        "protein_layout_instruction": protein_layout_instruction,
    }


def build_food_photo_prompt(item: dict) -> str:
    components = split_meal_components(item)
    return FOOD_PHOTO_PROMPT_TEMPLATE.format(**components)


def extract_generated_image_bytes(response) -> bytes:
    # Try generated_images style first
    generated_images = getattr(response, "generated_images", None) or []
    for generated in generated_images:
        image_obj = getattr(generated, "image", None)
        image_bytes = getattr(image_obj, "image_bytes", None) if image_obj else None
        if image_bytes:
            return image_bytes

    # Try candidates/parts inline data fallback
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                raw = inline.data
                if isinstance(raw, bytes):
                    return raw
                try:
                    return base64.b64decode(raw)
                except Exception:
                    pass

    return b""


def is_model_not_found_error(err_text: str) -> bool:
    text = (err_text or "").lower()
    return "404" in text and "not_found" in text


def is_retryable_error(err_text: str) -> bool:
    text = (err_text or "").lower()
    retry_markers = [
        "429",
        "resource_exhausted",
        "quota",
        "too many requests",
        "rate limit",
        "temporarily unavailable",
    ]
    return any(marker in text for marker in retry_markers)


def download_stock_food_photo(item: dict, output_path: Path):
    seed = abs(hash(str(item.get("name", "special-meal")))) % 100000
    url = f"https://loremflickr.com/1280/960/food,meal?lock={seed}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize to PNG so extension/content match for static hosting.
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image.save(output_path, format="PNG", optimize=True)


def generate_food_photo_image(item: dict, output_path: Path, image_api_key: str):
    from google import genai
    from google.genai import types

    if not image_api_key:
        raise ValueError("GEMINI_API_KEY_NANO (or GEMINI_API_KEY) environment variable not set")

    prompt = build_food_photo_prompt(item)
    client = genai.Client(api_key=image_api_key)

    def generate_with_model(model_name: str):
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=build_generate_config(response_modalities=["TEXT", "IMAGE"]),
        )

    delay = 8
    last_error = None
    for attempt in range(1, IMAGE_GEN_MAX_RETRIES + 1):
        should_retry = False
        for model_name in IMAGE_MODEL_CANDIDATES:
            try:
                response = generate_with_model(model_name)
                image_bytes = extract_generated_image_bytes(response)
                if not image_bytes:
                    raise RuntimeError(f"Model {model_name} returned no image bytes")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("wb") as fh:
                    fh.write(image_bytes)
                print(f"Generated AI image with model: {model_name}", file=sys.stderr)
                return
            except Exception as err:
                last_error = err
                err_text = str(err)
                if is_model_not_found_error(err_text):
                    print(f"Image model unavailable: {model_name}", file=sys.stderr)
                    continue
                if is_retryable_error(err_text):
                    should_retry = True
                    break
        if attempt >= IMAGE_GEN_MAX_RETRIES or not should_retry:
            break
        print(
            f"Image generation retry {attempt}/{IMAGE_GEN_MAX_RETRIES} after retryable error: {last_error}",
            file=sys.stderr,
        )
        time.sleep(delay)
        delay *= 2

    try:
        print("All configured image models failed; using stock food photo fallback.", file=sys.stderr)
        download_stock_food_photo(item, output_path)
        return
    except Exception:
        pass

    raise RuntimeError(f"Image generation failed: {last_error}")


def generate_or_reuse_food_photo_image(category: str, item: dict, output_path: Path, image_api_key: str, assets_dir: Path):
    if output_path.exists():
        print(f"Using existing generated image: {to_repo_relative_url(output_path)}", file=sys.stderr)
        return

    reusable_paths = get_reusable_image_paths(category, item, assets_dir)
    for reusable_path in reusable_paths:
        if copy_reusable_image(reusable_path, output_path):
            return

    generate_food_photo_image(item, output_path, image_api_key)

    if output_path.exists():
        reusable_path = reusable_paths[0]
        reusable_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_path, reusable_path)
        print(f"Archived menu image for future reuse: {to_repo_relative_url(reusable_path)}", file=sys.stderr)


def to_repo_relative_url(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def generate_daily_highlights(menu_data: dict, assets_dir: Path, week_key: str, image_api_key: str, target_days=None):
    generated = {}
    days_to_generate = [day for day in REQUIRED_DAYS if not target_days or day in target_days]
    for day in days_to_generate:
        day_menu = menu_data.get(day)
        if not isinstance(day_menu, dict) or not day_menu:
            continue
        if day == "Tuesday":
            category = TUESDAY_TACO_SPECIAL["category"]
            item = dict(TUESDAY_TACO_SPECIAL["item"])
            filename = f"daily-{week_key}-{slugify(day)}-{slugify(category)}.png"
            output_path = assets_dir / filename

            try:
                if TUESDAY_REFERENCE_IMAGE.exists():
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(TUESDAY_REFERENCE_IMAGE, output_path)
                    print(f"Using fixed Tuesday reference image: {to_repo_relative_url(TUESDAY_REFERENCE_IMAGE)}", file=sys.stderr)
                else:
                    generate_or_reuse_food_photo_image(category, item, output_path, image_api_key, assets_dir)
            except Exception as err:
                print(f"Daily image generation failed for {day}/{category}: {err}; using fallback image.", file=sys.stderr)
                render_fallback_tray_image(output_path)

            image_url = to_repo_relative_url(output_path)
            generated[day] = [{
                "category": category,
                "name": str(item.get("name") or ""),
                "imageUrl": image_url,
            }]
            print(f"Generated daily highlight: {day} / {category} -> {image_url}", file=sys.stderr)
            continue

        highlights = select_daily_highlights(day_menu)
        if not highlights:
            continue

        generated[day] = []
        for category, item in highlights:
            if not category or not isinstance(item, dict):
                continue
            filename = f"daily-{week_key}-{slugify(day)}-{slugify(category)}.png"
            output_path = assets_dir / filename

            try:
                generate_or_reuse_food_photo_image(category, item, output_path, image_api_key, assets_dir)
            except Exception as err:
                print(f"Daily image generation failed for {day}/{category}: {err}; using fallback image.", file=sys.stderr)
                render_fallback_tray_image(output_path)

            image_url = to_repo_relative_url(output_path)
            item["imageUrl"] = image_url
            generated[day].append({
                "category": category,
                "name": str(item.get("name") or ""),
                "imageUrl": image_url,
            })
            print(f"Generated daily highlight: {day} / {category} -> {image_url}", file=sys.stderr)

    return generated


def generate_weekly_highlights(menu_data: dict, assets_dir: Path, week_key: str, image_api_key: str):
    weekly = {}
    for category in CATEGORY_PRIORITY:
        representative_item = None
        for day in REQUIRED_DAYS:
            day_menu = menu_data.get(day)
            if not isinstance(day_menu, dict):
                continue
            _, item = find_category_entry(day_menu, category)
            if isinstance(item, dict):
                representative_item = item
                break

        if not representative_item:
            continue

        filename = f"weekly-{week_key}-{slugify(category)}.png"
        output_path = assets_dir / filename

        try:
            generate_or_reuse_food_photo_image(category, representative_item, output_path, image_api_key, assets_dir)
        except Exception as err:
            print(f"Weekly image generation failed for {category}: {err}; using fallback image.", file=sys.stderr)
            render_fallback_tray_image(output_path)

        weekly[category] = to_repo_relative_url(output_path)
        print(f"Generated weekly highlight: {category} -> {weekly[category]}", file=sys.stderr)

    return weekly


def normalize_daily_highlights(daily: dict) -> dict:
    if not isinstance(daily, dict):
        return {}
    normalized = {}
    for day, highlights in daily.items():
        if not isinstance(highlights, list):
            continue
        normalized_items = []
        for entry in highlights:
            if not isinstance(entry, dict):
                continue
            normalized_entry = dict(entry)
            parsed_name = parse_stringified_dict(normalized_entry.get("name"))
            if parsed_name and parsed_name.get("name"):
                normalized_entry["name"] = str(parsed_name.get("name")).strip()
            normalized_items.append(normalized_entry)
        if normalized_items:
            normalized[day] = normalized_items
    return normalized


def merge_daily_highlights(existing_daily: dict, generated_daily: dict, target_days=None) -> dict:
    if target_days:
        merged = dict(existing_daily) if isinstance(existing_daily, dict) else {}
        for day in target_days:
            if day in generated_daily:
                merged[day] = generated_daily[day]
            else:
                merged.pop(day, None)
        return merged
    return generated_daily


def load_existing_menu(menu_path: Path) -> dict:
    if not menu_path.exists():
        raise FileNotFoundError(f"Menu file not found: {menu_path}")
    with menu_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_existing_menu_if_present(menu_path: Path) -> dict:
    if not menu_path.exists():
        return {}
    return load_existing_menu(menu_path)


def write_menu(menu_path: Path, menu_data: dict):
    with menu_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(menu_data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def strip_volatile_metadata(value):
    if isinstance(value, dict):
        stripped = {}
        for key, item in value.items():
            if key in VOLATILE_METADATA_KEYS:
                continue
            stripped[key] = strip_volatile_metadata(item)
        return stripped
    if isinstance(value, list):
        return [strip_volatile_metadata(item) for item in value]
    return value


def has_meaningful_changes(original: dict, updated: dict) -> bool:
    return strip_volatile_metadata(original or {}) != strip_volatile_metadata(updated or {})


def parse_args():
    parser = argparse.ArgumentParser(description="Extract cafe menu and generate highlight imagery.")
    parser.add_argument("--menu-path", default=str(DEFAULT_MENU_PATH), help="Path to menu.json")
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS_DIR), help="Directory for generated images")
    parser.add_argument("--daily", action="store_true", help="Reuse menu.json and update only today's generated image metadata")
    parser.add_argument("--weekly", action="store_true", help="Refresh the source menu once, then update weekly metadata and today's image")
    parser.add_argument("--no-fetch", action="store_true", help="Skip remote extraction and reuse existing menu JSON as input")
    parser.add_argument("--skip-images", action="store_true", help="Run extraction/update without generating images")
    parser.add_argument("--today-only", action="store_true", help="Generate image(s) only for today's weekday special item")
    parser.add_argument("--all-daily-images", action="store_true", help="Generate daily images for every weekday instead of only today")
    parser.add_argument("--skip-weekly-images", action="store_true", help="Preserve existing weekly highlight images instead of regenerating them")
    parser.add_argument("--stdout", action="store_true", help="Print final menu JSON to stdout after writing to file")
    args = parser.parse_args()
    if args.daily and args.weekly:
        parser.error("--daily and --weekly cannot be used together")
    if args.daily:
        args.no_fetch = True
        args.today_only = True
        args.skip_weekly_images = True
    if args.weekly:
        args.no_fetch = False
        args.today_only = not args.all_daily_images
    return args


def get_current_day_name():
    try:
        return datetime.now(ZoneInfo(MENU_TIMEZONE)).strftime("%A")
    except Exception:
        # Fallback for environments without tzdata.
        return datetime.now().strftime("%A")


def main():
    args = parse_args()
    menu_path = Path(args.menu_path).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    week_key = datetime.now().strftime("%Yw%W")
    current_day_name = get_current_day_name()
    target_days = {current_day_name} if args.today_only else set(REQUIRED_DAYS)

    try:
        image_api_key = get_image_generation_api_key()
        if not image_api_key and not args.skip_images:
            print("Warning: GEMINI_API_KEY_NANO not set; image generation may fall back.", file=sys.stderr)

        existing_menu_data = load_existing_menu_if_present(menu_path)
        original_menu_data = json.loads(json.dumps(existing_menu_data)) if existing_menu_data else {}
        existing_metadata = existing_menu_data.get("_generated", {}) if isinstance(existing_menu_data, dict) else {}

        source_hash = None
        source_mime_type = None
        extraction_model_used = existing_metadata.get("extractionModel")

        if args.no_fetch:
            print("Loading existing menu JSON...", file=sys.stderr)
            if not existing_menu_data:
                raise FileNotFoundError(f"Menu file not found: {menu_path}")
            menu_data = existing_menu_data
        else:
            source_bytes, source_mime_type, source_hash = fetch_menu_source()
            if source_hash and source_hash == existing_metadata.get("sourceHash") and existing_menu_data:
                print("Menu source unchanged; reusing existing structured menu JSON.", file=sys.stderr)
                menu_data = existing_menu_data
            else:
                raw_json, extraction_model_used = extract_menu_with_gemini(source_bytes, source_mime_type)
                menu_data = json.loads(raw_json)

        menu_data = validate_menu_json(menu_data)
        metadata = menu_data.setdefault("_generated", {})
        existing_daily = normalize_daily_highlights(existing_metadata.get("dailyHighlights", {}))
        existing_weekly = existing_metadata.get("weeklyHighlights", {})

        if args.skip_images:
            daily = existing_daily or metadata.get("dailyHighlights") or {}
            weekly = metadata.get("weeklyHighlights") or existing_weekly or {}
            print("Skipping image generation by request (--skip-images).", file=sys.stderr)
        else:
            generated_daily = generate_daily_highlights(menu_data, assets_dir, week_key, image_api_key, target_days=target_days)
            daily = merge_daily_highlights(existing_daily, generated_daily, target_days if args.today_only else None)

            if args.skip_weekly_images or (args.today_only and not args.weekly):
                weekly = existing_weekly or metadata.get("weeklyHighlights") or {}
                print("Preserving existing weekly highlights.", file=sys.stderr)
            else:
                weekly = generate_weekly_highlights(menu_data, assets_dir, week_key, image_api_key)

        metadata["priorityOrder"] = CATEGORY_PRIORITY
        metadata["dailyHighlights"] = daily
        metadata["weeklyHighlights"] = weekly
        if source_hash:
            metadata["sourceUrl"] = MENU_SOURCE_URL
            metadata["sourceType"] = source_mime_type
            metadata["sourceHash"] = source_hash
            metadata["sourceFetchedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if extraction_model_used:
            metadata["extractionModel"] = extraction_model_used

        if has_meaningful_changes(original_menu_data, menu_data):
            metadata["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            write_menu(menu_path, menu_data)
            print(f"Wrote menu JSON with image URLs: {menu_path}", file=sys.stderr)
        else:
            print("No meaningful menu JSON changes; leaving menu.json unchanged.", file=sys.stderr)

        if args.stdout:
            print(json.dumps(menu_data, indent=2, ensure_ascii=False))

        return 0
    except requests.RequestException as err:
        print(f"Error fetching menu image: {err}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as err:
        print(f"Error parsing menu JSON: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
