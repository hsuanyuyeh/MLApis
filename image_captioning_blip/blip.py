from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

if __name__ == "__main__":
    # try blip base captioning model
    # load BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # load an image
    image = Image.open('images/sample_1.png')

    # prepare the image
    inputs = processor(image, return_tensors="pt")

    # generate captions
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    print("Generate Caption:", caption)
    
    # try blip large captioning model
    # load BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # image url
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    # specigy the question want to ask about the image
    question = "What is in the image?"

    # use the processor to prepare inputs for VQA (image + question)
    inputs = processor(raw_image, question, return_tensors="pt")

    # generate the answer from the model
    outputs = model.generate(**inputs)

    # decode and print the answer to the question
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"Answer: {answer}")