import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

url = "https://en.wikipedia.org/wiki/IBM"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

img_elements = soup.find_all("img")

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

with open("captions.txt", "w") as caption_file:
    for img_element in img_elements:
        img_url = img_element.get('src')

        # Skip if the images is an SVG or too small (likely an icon)
        if 'svg' in img_url or '1x1' in img_url:
            continue

        # Correct the URL if it's malformed
        if img_url.startswith("//"):
            img_url = 'https:' + img_url
        elif not img_url.startswith("http://") and not img_url.startswith("https://"):
            continue

        try:
            response = requests.get(img_url)
            raw_image = Image.open(BytesIO(response.content))
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue
            
            raw_image = raw_image.convert("RGB")
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{img_url}: {caption}\n")

        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

