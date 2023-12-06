import torch
from utils import img_to_tensor, tensor_to_img, load_image
import os
import transformer
import cv2


def stylize(style_path, content_folder, save_folder):

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    net = transformer.TransformerNet()
    net.load_state_dict(torch.load(style_path))
    net = net.to(device)

    # Stylize per frame
    images = [img for img in os.listdir(content_folder) if img.endswith(".jpg")]
    with torch.no_grad():
        for image_name in images:

            torch.cuda.empty_cache()
            
            # Load content image to tensor
            content_image = load_image(content_folder + image_name)
            content_tensor = img_to_tensor(content_image).to(device)

            # Compute generated image and convert it back to image
            generated_tensor = net(content_tensor)
            generated_image = tensor_to_img(generated_tensor.detach())
            generated_image = generated_image[:, :, [2, 1, 0]]

            cv2.imwrite(save_folder+ image_name + "_output.jpg", generated_image.clip(0, 255))

