#!/usr/bin/env python3
"""
Automated Cafe Menu Extractor
Fetches menu image from cafe website and converts to JSON using Gemini Vision API.
"""

import os
import sys
import json
import requests
import google.generativeai as genai
from PIL import Image
from io import BytesIO

MENU_IMAGE_URL = "https://cafe.sebastians.com/sebclients/3130alewife.jpg"

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
- price: The price(s) listed (use format like "$8.95" or "$2.90–$4.95" for ranges)

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


def fetch_menu_image():
    """Download the menu image from the cafe website."""
    print(f"Fetching menu image from {MENU_IMAGE_URL}...", file=sys.stderr)
    response = requests.get(MENU_IMAGE_URL, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    print(f"Image downloaded: {image.size[0]}x{image.size[1]}", file=sys.stderr)
    return image


def extract_menu_with_gemini(image):
    """Use Gemini Vision API to extract menu text and convert to JSON."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    print("Sending image to Gemini API...", file=sys.stderr)
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([EXTRACTION_PROMPT, image])

    raw_text = response.text.strip()

    # Clean up response - remove markdown code blocks if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.startswith("```")]
        raw_text = "\n".join(lines)

    return raw_text


def validate_menu_json(menu_json):
    """Validate the extracted menu JSON has the expected structure."""
    required_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for day in required_days:
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
            # Ensure all required fields exist
            item.setdefault("description", "")
            item.setdefault("price", "")

    return menu_json


def main():
    try:
        # Fetch the menu image
        image = fetch_menu_image()

        # Extract menu using Gemini
        raw_json = extract_menu_with_gemini(image)

        # Parse and validate
        menu_data = json.loads(raw_json)
        menu_data = validate_menu_json(menu_data)

        # Output formatted JSON to stdout
        print(json.dumps(menu_data, indent=2))

        # Print summary to stderr (visible in logs but not in output)
        print("\n=== Extraction Summary ===", file=sys.stderr)
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            if day in menu_data:
                categories = list(menu_data[day].keys())
                print(f"{day}: {', '.join(categories)}", file=sys.stderr)

        return 0

    except requests.RequestException as e:
        print(f"Error fetching menu image: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing menu JSON: {e}", file=sys.stderr)
        print(f"Raw response was: {raw_json[:500]}...", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
