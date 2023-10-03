# https://github.com/continue-revolution/sd-webui-segment-anything/wiki/API
import requests
url = "http://150.158.42.196:7861/sam/"
# url = "http://54.190.16.4:7860/sam/"

def samHeartBeat():
    response = requests.get(url + "heartbeat")
    reply = response.json()
    return(reply["msg"])

# print(samHeartBeat())

##############################################

def samAvaliableModels():
    response = requests.get(url + "sam-model")
    reply = response.json()
    return(reply)

# print(samAvaliableModels())

##############################################

import base64
from PIL import Image
from io import BytesIO

def filename_to_base64(filename):
    with open(filename, "rb") as fh:
        return base64.b64encode(fh.read())


def paste(img, row, grid):
    for idx, img in enumerate(img):
        img_pil = Image.open(BytesIO(base64.b64decode(img))).resize((512, 512))
        grid.paste(img_pil, (idx * 512, row * 512))


img_filename = "img_atm.png"


def samPredict_Dots(img, positive_points, negative_points):
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
    # print(reply["msg"])
    return reply


def run_samPredict_Dots(img, positive_points, negative_points):
    reply = samPredict_Dots(img, positive_points, negative_points)
    grid = Image.new('RGBA', (3 * 512, 3 * 512))

    print(f"Len of masks: {len(reply['masks'])}")

    paste(reply["blended_images"], 0, grid)
    paste(reply["masks"], 1, grid)
    paste(reply["masked_images"], 2, grid)
    grid.show()

# run_samPredict_Dots()

############################################
def samPredict_GroundingDINO(img):
    payload = {
        "sam_model_name": "sam_vit_h_4b8939.pth",
        "input_image": filename_to_base64(img).decode(),
        "dino_enabled": True,
        "dino_text_prompt": "the building in the middle",
        "dino_preview_checkbox": False,
    }
    response = requests.post(url + "sam-predict", json=payload)
    reply = response.json()
    # print(reply["msg"])
    # print(reply)
    return reply

def run_samPredict_GroundingDINO():
    reply = samPredict_GroundingDINO(img_filename)
    grid = Image.new('RGBA', (3 * 512, 3 * 512))

    paste(reply["blended_images"], 0, grid)
    paste(reply["masks"], 1, grid)
    paste(reply["masked_images"], 2, grid)
    grid.show()

# run_samPredict_GroundingDINO()

##########################################

def dinoPredict(img):
    payload = {
        "dino_model_name": "GroundingDINO_SwinT_OGC (694MB)",
        "input_image": filename_to_base64(img).decode(),
        "text_prompt": "the building in the middle",
    }
    response = requests.post(url + "dino-predict", json=payload)
    reply = response.json()
    # print(reply["msg"])
    # print(reply)
    return reply

def run_dinoPredict():
    reply = dinoPredict(img_filename)
    grid = Image.new('RGBA', (512, 512))
    paste([reply["image_with_box"]], 0, grid)
    grid.show()

# run_dinoPredict()

##################################


def dilateMask(img, mask):
    payload = {
        "input_image": filename_to_base64(img).decode(),
        "mask": mask
    }
    response = requests.post(url + "dilate-mask", json=payload)
    reply = response.json()
    # print(reply["msg"])
    return reply


def run_dilateMask():
    reply_GroundingDINO = samPredict_GroundingDINO(img_filename)
    # grid = Image.new('RGBA', (3 * 512, 4 * 512))

    # paste(reply_GroundingDINO["blended_images"], 0, grid)
    # paste(reply_GroundingDINO["masks"], 1, grid)
    # paste(reply_GroundingDINO["masked_images"], 2, grid)


    reply = dilateMask(img_filename, reply_GroundingDINO["masks"][0])

    # paste([reply["blended_image"], reply["mask"], reply["masked_image"]], 4, grid)
    # grid.show()
    grid = Image.new('RGBA', (3 * 512, 2 * 512))
    paste([reply_GroundingDINO["blended_images"][0], reply_GroundingDINO["masks"][0], reply_GroundingDINO["masked_images"][0]], 0, grid)
    paste([reply["blended_image"], reply["mask"], reply["masked_image"]], 1, grid)
    grid.show()


# run_dilateMask()

#################################

def controlnetSeg(img):
    payload = {
        "input_image": filename_to_base64(img).decode(),
        "sam_model_name": "sam_vit_l_0b3195.pth", # Optional ['sam_hq_vit_l.pth', 'sam_vit_h_4b8939.pth', 'mobile_sam.pt', 'sam_hq_vit_h.pth', 'sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_hq_vit_b.pth']
        "processor": "seg_ofade20k", # Optional
        # "processor": "seg_ufade20k",
        # "processor": "seg_ofcoco", #!!! out of VRAM, torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB (GPU 0; 14.61 GiB total capacity; 13.20 GiB already allocated; 851.56 MiB free; 13.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
        # "processor": "random",

    }
    response = requests.post(url + "controlnet-seg", json={
        "payload": payload,
        "autosam_conf": {
            # defaults:
            "points_per_side": 32, # Optional[int] = 32
            "points_per_batch": 64, #
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "stability_score_offset": 1.0,
            "box_nms_thresh": 0.7,
            "crop_n_layers": 0,
            "crop_nms_thresh": 0.7,
            "crop_overlap_ratio": 512 / 1500,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 0,
        }
    })

    reply = response.json()
    # print(reply["msg"])
    return reply

def run_controlnetSeg():
    reply = controlnetSeg(img_filename)

    grid = Image.new('RGBA', (2 * 512, 2 * 512))
    for key in reply.keys():
        print(key)
    if 'error' in reply.keys():
        print(f"{reply['error']}")
    if 'errors' in reply.keys():
        print(f"{reply['errors']}")
    paste([reply["blended_presam"], reply["blended_postsam"]], 0, grid)
    paste([reply["sem_presam"], reply["sem_postsam"]], 1, grid)
    grid.show()

# run_controlnetSeg()

################################

def controlnetSegRandom(img):
    payload = {
        "input_image": filename_to_base64(img).decode(),
        "sam_model_name": "sam_vit_l_0b3195.pth", # Optional ['sam_hq_vit_l.pth', 'sam_vit_h_4b8939.pth', 'mobile_sam.pt', 'sam_hq_vit_h.pth', 'sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_hq_vit_b.pth']
        # "processor": "seg_ofade20k", # Optional
        # "processor": "seg_ufade20k",
        # "processor": "seg_ofcoco", #!!! out of VRAM, torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB (GPU 0; 14.61 GiB total capacity; 13.20 GiB already allocated; 851.56 MiB free; 13.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
        "processor": "random",

    }
    response = requests.post(url + "controlnet-seg", json={
        "payload": payload,
        "autosam_conf": {
            # defaults:
            "points_per_side": 32, # Optional[int] = 32
            "points_per_batch": 64, #
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "stability_score_offset": 1.0,
            "box_nms_thresh": 0.7,
            "crop_n_layers": 0,
            "crop_nms_thresh": 0.7,
            "crop_overlap_ratio": 512 / 1500,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 0,
        }
    })

    reply = response.json()
    # print(reply["msg"])
    return reply

def run_controlnetSegRandom():
    reply = controlnetSegRandom(img_filename)

    grid = Image.new('RGBA', (2 * 512, 2 * 512))
    for key in reply.keys():
        print(key)
    if 'error' in reply.keys():
        print(f"{reply['error']}")
    if 'errors' in reply.keys():
        print(f"{reply['errors']}")
    paste([reply["blended_images"], reply["random_seg"]], 0, grid)
    paste([reply["edit_anything_control"], ], 1, grid)
    # Left above (0) is blended image, right above (1) is random segmentation, left below (2) is Edit-Anything control input.
    grid.show()

# run_controlnetSegRandom()

#################################

print(samHeartBeat())
print(samAvaliableModels())
positive_points = [
    [100, 150],  # Coordinates of a positive point in the image
    [300, 250],  # Coordinates of another positive point
]

negative_points = [
    [200, 200],  # Coordinates of a negative point in the image
]
run_samPredict_Dots("img_atm.png", positive_points, negative_points)



run_samPredict_GroundingDINO()
run_dilateMask()
run_controlnetSeg()
run_controlnetSegRandom()