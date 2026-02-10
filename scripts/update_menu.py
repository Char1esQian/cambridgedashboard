#!/usr/bin/env python3
"""
Automated cafe menu updater.

Workflow:
1) Fetch menu image and extract structured JSON with Gemini Vision (default), or reuse existing menu.json.
2) Generate daily and weekly highlight images for priority stations.
3) Save generated images under assets/menu-generated/.
4) Write image URLs back into menu.json so the dashboard can render them.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

MENU_IMAGE_URL = "https://cafe.sebastians.com/sebclients/3130alewife.jpg"
REQUIRED_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CATEGORY_PRIORITY = ["Carving", "Plant Power", "Action"]
DEFAULT_CANVAS = (1280, 720)

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

Return ONLY valid JSON in this exact format, no markdown code blocks:
{
  "Monday": {
    "Breakfast": {"name": "...", "description": "...", "price": "..."},
    "Soup": {"name": "...", "description": "...", "price": "..."},
    ...
  },
  "Tuesday": { ... },
  ...
}

Important:
- Include only weekdays (Monday-Friday)
- Use the exact category names as shown on the menu
- If a price is not visible, use "Market Price"
- Ensure all JSON is properly formatted with double quotes
"""


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized or "item"


def normalize_price(value: str) -> str:
    return str(value or "").replace("–", "-").replace("â€“", "-").strip()


def fetch_menu_image():
    print(f"Fetching menu image from {MENU_IMAGE_URL}...", file=sys.stderr)
    response = requests.get(MENU_IMAGE_URL, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    print(f"Image downloaded: {image.size[0]}x{image.size[1]}", file=sys.stderr)
    return image, response.content


def get_extraction_api_key() -> str:
    return os.environ.get("GEMINI_EXTRACTION_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def get_image_generation_api_key() -> str:
    return os.environ.get("GEMINI_IMAGE_API_KEY") or ""


def extract_menu_with_gemini(image_bytes: bytes) -> str:
    from google import genai
    from google.genai import types

    api_key = get_extraction_api_key()
    if not api_key:
        raise ValueError("GEMINI_EXTRACTION_API_KEY (or GEMINI_API_KEY) environment variable not set")

    print("Sending image to Gemini API...", file=sys.stderr)
    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[EXTRACTION_PROMPT, image_part],
    )
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        lines = [line for line in raw_text.split("\n") if not line.startswith("```")]
        raw_text = "\n".join(lines)
    return raw_text


def validate_menu_json(menu_json):
    for day in REQUIRED_DAYS:
        if day not in menu_json:
            print(f"Warning: Missing day '{day}' in menu", file=sys.stderr)
            continue
        day_menu = menu_json[day]
        if not isinstance(day_menu, dict):
            raise ValueError(f"Invalid menu structure for {day}")
        for category, item in day_menu.items():
            if not isinstance(item, dict):
                raise ValueError(f"Invalid item structure for {day}/{category}")
            if "name" not in item:
                raise ValueError(f"Missing 'name' for {day}/{category}")
            item.setdefault("description", "")
            item.setdefault("price", "")
            item["price"] = normalize_price(item["price"])
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
    first = next(((c, i) for c, i in day_menu.items() if isinstance(i, dict)), (None, None))
    return first


def choose_palette(category: str):
    key = str(category or "").lower()
    if "carving" in key:
        return ((64, 34, 24), (138, 68, 45), (236, 186, 146))
    if "plant" in key:
        return ((24, 70, 40), (47, 122, 73), (180, 231, 190))
    if "action" in key:
        return ((26, 44, 86), (47, 94, 173), (176, 206, 255))
    return ((33, 57, 84), (67, 114, 161), (197, 222, 247))


def load_fonts():
    try:
        return (
            ImageFont.truetype("arialbd.ttf", 64),
            ImageFont.truetype("arial.ttf", 38),
            ImageFont.truetype("arialbd.ttf", 30),
            ImageFont.truetype("arial.ttf", 26),
        )
    except OSError:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback, fallback


def draw_wrapped_text(draw, text, font, fill, box, line_spacing=6):
    x, y, w, h = box
    words = str(text or "").split()
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), trial, font=font)[2]
        if width <= w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    cursor_y = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_h = bbox[3] - bbox[1]
        if cursor_y + line_h > y + h:
            break
        draw.text((x, cursor_y), line, font=font, fill=fill)
        cursor_y += line_h + line_spacing


def render_highlight_image(title: str, subtitle: str, detail: str, category: str, output_path: Path):
    width, height = DEFAULT_CANVAS
    base = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(base)
    c0, c1, c2 = choose_palette(category)

    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(c0[0] * (1 - t) + c1[0] * t)
        g = int(c0[1] * (1 - t) + c1[1] * t)
        b = int(c0[2] * (1 - t) + c1[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    draw.rounded_rectangle((60, 70, width - 60, height - 70), radius=34, fill=(255, 255, 255, 230))
    draw.rounded_rectangle((96, 108, 420, 164), radius=18, fill=c2)

    font_title, font_subtitle, font_pill, font_detail = load_fonts()
    draw.text((116, 120), category.upper(), font=font_pill, fill=(18, 30, 42))
    draw_wrapped_text(draw, title, font_title, (18, 30, 42), (100, 200, width - 200, 250), line_spacing=12)
    draw_wrapped_text(draw, subtitle, font_subtitle, (38, 59, 82), (100, 470, width - 200, 120), line_spacing=8)
    draw_wrapped_text(draw, detail, font_detail, (55, 77, 102), (100, 610, width - 200, 60), line_spacing=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(output_path, format="PNG", optimize=True)


def to_repo_relative_url(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return rel


def generate_daily_highlights(menu_data: dict, assets_dir: Path, week_key: str):
    generated = {}
    for day in REQUIRED_DAYS:
        day_menu = menu_data.get(day)
        if not isinstance(day_menu, dict) or not day_menu:
            continue
        category, item = select_daily_highlight(day_menu)
        if not category or not isinstance(item, dict):
            continue
        filename = f"daily-{week_key}-{slugify(day)}-{slugify(category)}.png"
        output_path = assets_dir / filename
        title = item.get("name", "Menu Highlight")
        subtitle = item.get("description", "") or "Freshly featured station item."
        detail = f"{day} highlight - {normalize_price(item.get('price', 'Market Price'))}"
        render_highlight_image(title, subtitle, detail, category, output_path)
        image_url = to_repo_relative_url(output_path)
        item["imageUrl"] = image_url
        generated[day] = {"category": category, "imageUrl": image_url}
        print(f"Generated daily highlight: {day} / {category} -> {image_url}", file=sys.stderr)
    return generated


def generate_weekly_highlights(menu_data: dict, assets_dir: Path, week_key: str):
    weekly = {}
    for category in CATEGORY_PRIORITY:
        lines = []
        for day in REQUIRED_DAYS:
            day_menu = menu_data.get(day)
            if not isinstance(day_menu, dict):
                continue
            _, item = find_category_entry(day_menu, category)
            if isinstance(item, dict):
                lines.append(f"{day}: {item.get('name', 'Menu item')}")
        if not lines:
            continue
        filename = f"weekly-{week_key}-{slugify(category)}.png"
        output_path = assets_dir / filename
        render_highlight_image(
            title=f"{category} Weekly Highlights",
            subtitle=" / ".join(lines[:2]),
            detail=" | ".join(lines[2:]) if len(lines) > 2 else f"Week of {week_key}",
            category=category,
            output_path=output_path,
        )
        weekly[category] = to_repo_relative_url(output_path)
        print(f"Generated weekly highlight: {category} -> {weekly[category]}", file=sys.stderr)
    return weekly


def load_existing_menu(menu_path: Path):
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
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip remote extraction and reuse existing menu JSON as input",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print final menu JSON to stdout after writing to file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    menu_path = Path(args.menu_path).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    week_key = datetime.now().strftime("%Yw%W")

    try:
        image_api_key = get_image_generation_api_key()
        if image_api_key:
            print("Detected GEMINI_IMAGE_API_KEY (reserved for future image API generation).", file=sys.stderr)

        if args.no_fetch:
            print("Loading existing menu JSON...", file=sys.stderr)
            menu_data = load_existing_menu(menu_path)
        else:
            _, image_bytes = fetch_menu_image()
            raw_json = extract_menu_with_gemini(image_bytes)
            menu_data = json.loads(raw_json)

        menu_data = validate_menu_json(menu_data)
        daily = generate_daily_highlights(menu_data, assets_dir, week_key)
        weekly = generate_weekly_highlights(menu_data, assets_dir, week_key)

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
