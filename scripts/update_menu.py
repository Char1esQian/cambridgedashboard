#!/usr/bin/env python3
"""
Automated cafe menu updater.

Workflow:
1) Fetch menu image and extract structured JSON with Gemini Vision (default), or reuse existing menu.json.
2) Generate daily and weekly food photos for priority stations.
3) Save generated images under assets/menu-generated/.
4) Write image URLs back into menu.json so the dashboard can render them.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw

MENU_IMAGE_URL = "https://cafe.sebastians.com/sebclients/3130alewife.jpg"
REQUIRED_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CATEGORY_PRIORITY = ["Carving", "Plant Power", "Action"]
DEFAULT_CANVAS = (1280, 720)
IMAGE_MODEL = "nano-banana-pro"
IMAGE_GEN_MAX_RETRIES = 4
MENU_TIMEZONE = "America/New_York"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MENU_PATH = PROJECT_ROOT / "menu.json"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets" / "menu-generated"

EXTRACTION_PROMPT = """Analyze this cafe menu image and extract ALL menu items into a structured JSON format.

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

FOOD_PHOTO_PROMPT_TEMPLATE = """Photoreal food product photo: a single black rectangular meal-prep container with a clear lid slightly open, on a clean off-white table. Modern cafeteria background, heavily blurred chairs, soft daylight from left, shallow depth of field, minimal/neutral colors, lots of negative space, 3/4 table-height angle, crisp focus on food.

Meal based on: {item_name}. Ingredients: {ingredients_list}. Base/crust: {base_or_crust}. Sauce: {sauce_or_none}. Make it realistic, neatly arranged inside the tray, appetizing, no props.

Negative: text, logos, labels, watermarks, hands/people, utensils, clutter, cartoon/anime, oversaturated, distorted container/lid.
"""


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized or "item"


def normalize_price(value: str) -> str:
    return str(value or "").replace("\u2013", "-").replace("â€“", "-").strip()


def get_extraction_api_key() -> str:
    return os.environ.get("GEMINI_EXTRACTION_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def get_image_generation_api_key() -> str:
    return os.environ.get("GEMINI_IMAGE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def fetch_menu_image() -> tuple[Image.Image, bytes]:
    print(f"Fetching menu image from {MENU_IMAGE_URL}...", file=sys.stderr)
    response = requests.get(MENU_IMAGE_URL, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    print(f"Image downloaded: {image.size[0]}x{image.size[1]}", file=sys.stderr)
    return image, response.content


def extract_menu_with_gemini(image_bytes: bytes) -> str:
    from google import genai
    from google.genai import types

    api_key = get_extraction_api_key()
    if not api_key:
        raise ValueError("GEMINI_EXTRACTION_API_KEY (or GEMINI_API_KEY) environment variable not set")

    print("Sending menu image to Gemini extraction API...", file=sys.stderr)
    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[EXTRACTION_PROMPT, image_part],
    )

    raw_text = (response.text or "").strip()
    if raw_text.startswith("```"):
        lines = [line for line in raw_text.split("\n") if not line.startswith("```")]
        raw_text = "\n".join(lines)
    return raw_text


def validate_menu_json(menu_json: dict) -> dict:
    def coerce_menu_item(item):
        if isinstance(item, dict):
            name = item.get("name") or item.get("title") or item.get("item") or item.get("dish")
            if not name:
                return None
            return {
                "name": str(name).strip(),
                "description": str(item.get("description") or item.get("details") or "").strip(),
                "price": normalize_price(item.get("price") or item.get("cost") or "Market Price"),
            }
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
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


def select_daily_highlight(day_menu: dict):
    for category in CATEGORY_PRIORITY:
        matched_category, item = find_category_entry(day_menu, category)
        if matched_category and item:
            return matched_category, item
    return next(((c, i) for c, i in day_menu.items() if isinstance(i, dict)), (None, None))


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


def split_meal_components(item: dict) -> dict:
    name = str(item.get("name", "")).strip()
    description = str(item.get("description", "")).strip()
    raw = ", ".join(part for part in [name, description] if part)
    tokens = [t.strip() for t in raw.split(",") if t.strip()]

    base_keywords = ("rice", "noodle", "pasta", "crust", "flatbread", "quinoa", "potato")
    sauce_keywords = ("sauce", "aioli", "pesto", "gravy", "chimichurri", "marinara", "curry", "dressing")

    base = "none"
    sauce = "none"
    ingredients = []

    for token in tokens:
        lower = token.lower()
        if base == "none" and any(k in lower for k in base_keywords):
            base = token
            continue
        if sauce == "none" and any(k in lower for k in sauce_keywords):
            sauce = token
            continue
        ingredients.append(token)

    if not ingredients and name:
        ingredients = [name]

    return {
        "item_name": name or "Chef special",
        "ingredients_list": ", ".join(ingredients) if ingredients else "Chef selected seasonal ingredients",
        "base_or_crust": base,
        "sauce_or_none": sauce,
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


def generate_food_photo_image(item: dict, output_path: Path, image_api_key: str):
    from google import genai
    from google.genai import types

    if not image_api_key:
        raise ValueError("GEMINI_IMAGE_API_KEY (or GEMINI_API_KEY) environment variable not set")

    prompt = build_food_photo_prompt(item)
    client = genai.Client(api_key=image_api_key)

    delay = 8
    last_error = None
    for attempt in range(1, IMAGE_GEN_MAX_RETRIES + 1):
        try:
            response = client.models.generate_images(
                model=IMAGE_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="4:3",
                ),
            )
            image_bytes = extract_generated_image_bytes(response)
            if not image_bytes:
                raise RuntimeError("Image model returned no image bytes")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as fh:
                fh.write(image_bytes)
            return
        except Exception as err:
            last_error = err
            err_text = str(err)
            retryable = "429" in err_text or "RESOURCE_EXHAUSTED" in err_text or "rate" in err_text.lower()
            if attempt >= IMAGE_GEN_MAX_RETRIES or not retryable:
                break
            print(f"Image generation retry {attempt}/{IMAGE_GEN_MAX_RETRIES} after error: {err}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Image generation failed: {last_error}")


def to_repo_relative_url(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def generate_daily_highlights(menu_data: dict, assets_dir: Path, week_key: str, image_api_key: str, target_days=None):
    generated = {}
    days_to_generate = [day for day in REQUIRED_DAYS if not target_days or day in target_days]
    for day in days_to_generate:
        day_menu = menu_data.get(day)
        if not isinstance(day_menu, dict) or not day_menu:
            continue
        category, item = select_daily_highlight(day_menu)
        if not category or not isinstance(item, dict):
            continue

        filename = f"daily-{week_key}-{slugify(day)}-{slugify(category)}.png"
        output_path = assets_dir / filename

        try:
            generate_food_photo_image(item, output_path, image_api_key)
        except Exception as err:
            print(f"Daily image generation failed for {day}/{category}: {err}; using fallback image.", file=sys.stderr)
            render_fallback_tray_image(output_path)

        image_url = to_repo_relative_url(output_path)
        item["imageUrl"] = image_url
        generated[day] = {"category": category, "imageUrl": image_url}
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
            generate_food_photo_image(representative_item, output_path, image_api_key)
        except Exception as err:
            print(f"Weekly image generation failed for {category}: {err}; using fallback image.", file=sys.stderr)
            render_fallback_tray_image(output_path)

        weekly[category] = to_repo_relative_url(output_path)
        print(f"Generated weekly highlight: {category} -> {weekly[category]}", file=sys.stderr)

    return weekly


def load_existing_menu(menu_path: Path) -> dict:
    if not menu_path.exists():
        raise FileNotFoundError(f"Menu file not found: {menu_path}")
    with menu_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_menu(menu_path: Path, menu_data: dict):
    with menu_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(menu_data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract cafe menu and generate highlight imagery.")
    parser.add_argument("--menu-path", default=str(DEFAULT_MENU_PATH), help="Path to menu.json")
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS_DIR), help="Directory for generated images")
    parser.add_argument("--no-fetch", action="store_true", help="Skip remote extraction and reuse existing menu JSON as input")
    parser.add_argument("--skip-images", action="store_true", help="Run extraction/update without generating images")
    parser.add_argument("--today-only", action="store_true", help="Generate image(s) only for today's weekday special item")
    parser.add_argument("--stdout", action="store_true", help="Print final menu JSON to stdout after writing to file")
    return parser.parse_args()


def main():
    args = parse_args()
    menu_path = Path(args.menu_path).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    week_key = datetime.now().strftime("%Yw%W")
    current_day_name = datetime.now(ZoneInfo(MENU_TIMEZONE)).strftime("%A")
    target_days = {current_day_name} if args.today_only else set(REQUIRED_DAYS)

    try:
        image_api_key = get_image_generation_api_key()
        if not image_api_key:
            print("Warning: GEMINI_IMAGE_API_KEY not set; using fallback local tray images.", file=sys.stderr)

        if args.no_fetch:
            print("Loading existing menu JSON...", file=sys.stderr)
            menu_data = load_existing_menu(menu_path)
        else:
            _, image_bytes = fetch_menu_image()
            raw_json = extract_menu_with_gemini(image_bytes)
            menu_data = json.loads(raw_json)

        menu_data = validate_menu_json(menu_data)
        if args.skip_images:
            daily = menu_data.get("_generated", {}).get("dailyHighlights", {})
            weekly = menu_data.get("_generated", {}).get("weeklyHighlights", {})
            print("Skipping image generation by request (--skip-images).", file=sys.stderr)
        else:
            daily = generate_daily_highlights(menu_data, assets_dir, week_key, image_api_key, target_days=target_days)
            weekly = {} if args.today_only else generate_weekly_highlights(menu_data, assets_dir, week_key, image_api_key)

        metadata = menu_data.setdefault("_generated", {})
        metadata["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        metadata["priorityOrder"] = CATEGORY_PRIORITY
        metadata["dailyHighlights"] = daily
        metadata["weeklyHighlights"] = weekly

        write_menu(menu_path, menu_data)
        print(f"Wrote menu JSON with image URLs: {menu_path}", file=sys.stderr)

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
