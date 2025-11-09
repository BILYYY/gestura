from pathlib import Path
from PIL import Image

REFS = Path("resources/")

for jpg_file in REFS.glob("*.jpg"):
    png_file = jpg_file.with_suffix(".png")
    img = Image.open(jpg_file)
    img.save(png_file, "PNG")
    print(f"Converted: {jpg_file.name} → {png_file.name}")
    jpg_file.unlink()  # Delete original .jpg

print("Done!")