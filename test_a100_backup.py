# https://github.com/continue-revolution/sd-webui-segment-anything/wiki/API
import requests
# url = "http://150.158.42.196:7860/sam/"
url = "http://34.223.252.10:7860/sam/"
def SamHeartBeat():
    response = requests.get(url + "heartbeat")
    reply = response.json()
    return(reply["msg"])

print(SamHeartBeat())

def SamAvaliableModels():
    response = requests.get(url + "sam-model")
    reply = response.json()
    return(reply)

print(SamAvaliableModels())


import base64
from PIL import Image, ImageDraw
from io import BytesIO

def filename_to_base64(filename):
    with open(filename, "rb") as fh:
        return base64.b64encode(fh.read())


def paste(img, row):
    def paste(img, row):
        for idx, img_data in enumerate(img):
            try:
                img_pil = Image.open(BytesIO(base64.b64decode(img_data))).resize((512, 512))
                grid.paste(img_pil, (idx * 512, row * 512))
            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")
                # Add additional debug information if needed


positive_points = [
        [100, 150],  # Coordinates of a positive point in the image
        [300, 250],  # Coordinates of another positive point
]

negative_points = [
    [200, 200],  # Coordinates of a negative point in the image
]

def SamPredict(img):
    payload = {
        "sam_model_name": "sam_vit_h_4b8939.pth",
        "input_image": filename_to_base64(img).decode(),
        "dino_enabled": False,
        # "dino_text_prompt": "the building in the middle",
        # "dino_preview_checkbox": False,
        "sam_positive_points": positive_points,
        "sam_negative_points": negative_points
    }
    response = requests.post(url + "sam-predict", json=payload)
    reply = response.json()
    print(reply["msg"])
    return reply

img_filename = "img_atm.png"
reply = SamPredict(img_filename)
#
# grid = Image.new('RGBA', (1 * 512, 1 * 512))
# paste(reply["blended_images"], 0)
# grid.show()
#
# grid = Image.new('RGBA', (1 * 512, 1 * 512))
# paste(reply["masks"], 0)
# grid.show()
#
# grid = Image.new('RGBA', (1 * 512, 1 * 512))
# paste(reply["masked_images"], 0)
# grid.show()
#
#
# grid = Image.new('RGBA', (3 * 512, 3 * 512))
#
# paste(reply["blended_images"], 0)
# paste(reply["masks"], 1)
# paste(reply["masked_images"], 2)
# grid.show()
#
grid = Image.new('RGBA', (3 * 512, 3 * 512))
#
paste([reply["blended_images"][0], reply["masks"][0], reply["masked_images"][0]], 0)
paste([reply["blended_images"][1], reply["masks"][1], reply["masked_images"][1]], 1)
paste([reply["blended_images"][2], reply["masks"][2], reply["masked_images"][2]], 2)
grid.show()

grid = Image.new('RGBA', (3 * 512, 3 * 512))

def paste_with_points(images, row, positive_points, negative_points):
    # Create a copy of the image to draw points on
    img_with_points = Image.open(BytesIO(base64.b64decode(images[0]))).resize((512, 512))

    # Create a drawing context
    draw = ImageDraw.Draw(img_with_points)

    # Draw positive points as black dots
    for point in positive_points:
        draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='black', outline='black')

    # Draw negative points as red dots
    for point in negative_points:
        draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='red', outline='red')

    # Paste the image with points onto the grid
    paste([img_with_points, images[1], images[2]], row)

# Call the updated paste function for each set of images
paste_with_points([reply["blended_images"][0], reply["masks"][0], reply["masked_images"][0]], 0, positive_points, negative_points)
paste_with_points([reply["blended_images"][1], reply["masks"][1], reply["masked_images"][1]], 1, positive_points, negative_points)
paste_with_points([reply["blended_images"][2], reply["masks"][2], reply["masked_images"][2]], 2, positive_points, negative_points)

# Show the grid
grid.show()