import pandas as pd
from pathlib import Path
import numpy as np

IN_PATH  = Path("04-data/combined/all_cars_combined.csv")
OUT_DIR  = Path("04-data/processed")
OUT_FILE = OUT_DIR / "cars_clean.csv"

# Mapping messy brand labels from filenames to canonical names
BRAND_MAP = {
    "vw": "volkswagen",
    "vauxhall": "vauxhall",
    "toyota": "toyota",
    "skoda": "skoda",
    "merc": "mercedes",
    "cclass": "mercedes",
    "bmw": "bmw",
    "audi": "audi",
    "ford": "ford",
    "focus": "ford",        # dataset is Ford Focus; brand should be ford
    "hyundi": "hyundai",    # misspelling in filename → fix
    "hyundai": "hyundai"
}

# Brand tiers
BRAND_TIER = {
    # luxury
    "audi": "luxury",
    "bmw": "luxury",
    "mercedes": "luxury",
    # midrange
    "volkswagen": "midrange",
    "skoda": "midrange",
    "toyota": "midrange",
    # economy
    "ford": "economy",
    "vauxhall": "economy",
    "hyundai": "economy",
}