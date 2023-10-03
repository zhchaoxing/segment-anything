import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import supervision as sv

# A workaround to handle mulitple copies of libiomp5md.dll
# Original error message: Error #15: Initializing libiomp5md.dll,
# but found libiomp5md.dll already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

image = cv2.imread("img_atm.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import time

# Start time
start_time = time.time()

# Set the device based on the availability of CUDA
TORCH_DEVICE = 'cpu'
if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda:0'

print(f"TORCH_DEVICE={TORCH_DEVICE}")
DEVICE = torch.device(TORCH_DEVICE)
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Select the model type and checkpoint path
# MODEL_TYPE = "vit_h"
# CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"

# MODEL_TYPE = "vit_l"
# CHECKPOINT_PATH = "models/sam_vit_l_0b3195.pth"

MODEL_TYPE = "vit_b"
# CHECKPOINT_PATH = "models/sam_vit_b_01ec64.pth"
CHECKPOINT_PATH = "models/sam_hq_vit_b.pth"


# Initialize the SAM model and redirect it
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Check how long it take
print("--- %s seconds ---" % (time.time() - start_time))

# Create a mask annotator
mask_annotator = sv.MaskAnnotator()

# Convert the masks to detections
detections = sv.Detections.from_sam(masks)

# Create annotated image with the mask and detections
annotated_image = mask_annotator.annotate(image, detections)

# Show the annotated image
plt.figure(figsize=(10,10))
plt.imshow(annotated_image)
plt.axis('on')
plt.show()