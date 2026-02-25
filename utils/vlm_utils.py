"""
Utility functions for VLM
"""
from openai import OpenAI

import os
import ast
import base64
import requests
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")


##############################################
## Text embedding
##############################################
def get_text_embedding_options(option="embeddings_oai"):
    """
    Get text embedding function based on the option
    """
    if option == "embeddings_oai":
        return get_text_embedding
    elif option == "embeddings_st":
        return get_text_embedding_sentence_transformer
    else:
        raise ValueError(f"Invalid option: {option}")

def get_text_embedding(text, model="text-embedding-3-large", dim=1024):
    """
    Get openai text embedding with specified dimension
    """
    
    client = OpenAI(
      base_url="https://api.gptsapi.net/v1",
      api_key="sk-CWP2517500a761fd50613af3577f70d63bfa0fdd322Pef5Z",
      max_retries=5, # 增加重试次数
      timeout=60.0
)
    embedding = np.array(client.embeddings.create(input=[text], model=model, dimensions=dim).data[0].embedding)
    return embedding

def get_text_embedding_sentence_transformer(text, model_name="all-MiniLM-L6-v2"):
    """
    Get text embedding with sentence transformer
    """
    model = SentenceTransformer(model_name)
    embedding = sentence_transformer_model.encode(text) # shape (D,)
    return embedding

##############################################
## Creating payloads for VLM
##############################################
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_payload_gpt4(messages):
    """
    Create payload for GPT-4.
    Input:
        messages: list of dicts, each in form {"role": str, "content": list of content dicts}
    """
    payload = {
        "model": "gpt-4o",
        "messages": messages,
    }
    return payload

def get_response(payload):
    client = OpenAI(
      base_url="https://api.gptsapi.net/v1",
      api_key="sk-CWP2517500a761fd50613af3577f70d63bfa0fdd322Pef5Z",
      max_retries=5, # 增加重试次数
      timeout=60.0
)
    return client.chat.completions.create(**payload)

def create_content_list(text_list, images, is_claude=False):
    """
    Create content list for gpt4v input.
    Input: 
        text_list: list of strings, text to be converted to content list; if starting with '#', it is an image reference
        images: dict, image name to image base64
        is_claude: bool, whether the content list is for claude
    Output:
        content_list: list of content dicts to be used for message content
    """
    content_list = []
    for text in text_list:
        content_list.append({"type": "text", "text": text})

    if images is not None:
        if is_claude:
            for img in images:
                content_list.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}})
        else:
            for img in images:
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

    return content_list

def query_model_text_only(system_prompt, user_prompt):
    """
    Util function to query the model with text only system and user prompts.
    Input:
        system_prompt: list of strings, system prompt
        user_prompt: list of strings, user prompt
    Output:
        text: str, response from the model
    """
    text_list = [' '.join(system_prompt), ' '.join(user_prompt)]
    content = create_content_list(text_list, [])
    payload = create_payload_gpt4([{"role":"system", "content":[content[0]]}, {"role":"user", "content":[content[1]]}])
    response = get_response(payload)
    if 'choices' not in response:
        print("Failed to get response.")
        print("Output detail", response)
        return None
    
    text = response['choices'][0]['message']['content']
    return text

def formulate_input(img_dir, prefix):
    """
    Creates image dict for prompt input.
    Input:
        img_dir: str, directory containing images original.png, proposal.png
        prefix: str, prefix for image names, serve as identifier in returned dict
    Output:
        encoded_imgs: dict of str, image name to base64 encoded image
        color_names: dict of str, image name to str, color names detected in proposals
    """
    def list_to_str(lst):
        return '[' + ', '.join(lst) + ']'

    names = ['original', 'proposal']
    encoded_imgs = []
    color_names = {}

    for name in names:
        # get image path
        img_path = os.path.join(img_dir, f'{prefix}_{name}.png')

        # encode image into base64
        encoded_imgs.append(encode_image(img_path))

        # find names for cluster colors used for proposal image(s)
        if 'proposal' in name:
            color_names[f'{prefix}_{name}'] = list_to_str(detect_colors_in_image(img_path))

    return encoded_imgs, color_names

###############################################
## Output parsing
###############################################
def parse_lm_output(output, parse_lst=True, parse_dict = False):
    """
    Parses the output of a language model to extract a list of strings following 'ANSWER:'.

    Parameters:
    output (str): The output string from a language model, expected to contain 'ANSWER: [...]'

    Returns:
    list: A Python list containing the elements found after 'ANSWER:', or None if the format is incorrect.
    """
    try:
        # Find the index where 'ANSWER:' appears in the output
        start_index = output.find('ANSWER:')
        if start_index == -1:
            # If 'ANSWER:' is not found, return None
            print("'ANSWER' is not found.\n")
            print(output)
            return None

        # Extract the substring starting from 'ANSWER:'
        answer_text = output[start_index + len('ANSWER:'):].strip()

        # check if the not identifiable flag is set 
        if 'cannot identify' in answer_text.lower():
            return None

        if parse_lst:
            # Use ast.literal_eval to safely evaluate the string as a Python expression
            answer_list = ast.literal_eval(answer_text)

            # Check if the result is indeed a list
            if isinstance(answer_list, list):
                return answer_list
            else:
                return None
            
        if parse_dict:
            # Use ast.literal_eval to safely evaluate the string as a Python expression
            answer_list = ast.literal_eval(answer_text)

            # Check if the result is indeed a list
            if isinstance(answer_list, dict):
                return answer_list
            else:
                return None
            
        else:
            return answer_text

        
    except:
        # In case of any error during parsing, return None
        print("Failed to parse: ")
        print(output)
        return None
    

###############################################
## Image related utils
###############################################

# pallette used in cluster visualization
PALETTE = [
    ([230, 25, 75], "Red"),
    ([60, 180, 75], "Green"),
    ([0, 130, 200], "Blue"),
    ([255, 225, 25], "Yellow"),
    ([245, 130, 48], "Orange"),
    ([145, 30, 180], "Purple"),
    ([70, 240, 240], "Cyan"),
    ([240, 50, 230], "Magenta"),
    ([250, 190, 212], "Pink"),
    ([210, 245, 60], "Lime Green"),
    ([0, 128, 128], "Teal"),
    ([170, 110, 40], "Brown"),
    ([128, 0, 0], "Maroon"),
    ([0, 0, 128], "Navy"),
    ([107, 142, 35], "Olive"),
    ([128, 128, 128], "Gray"),
    ([220, 20, 60], "Crimson"),
    ([0, 0, 0], "Black"),
    ([204, 85, 0], "Burnt Orange"),
    ([0, 153, 143], "Jade"),
    ]

def detect_colors_in_image(image_path):
    """
    Detects if an image contains specific colors.

    Parameters:
    image_path (str): Path to the image file.
    palette (list): A list of tuples, each containing an RGB color and its name, e.g., [([230, 25, 75], "Red"), ...].

    Returns:
    list: A list of color names that exist in the image (at least 5 pixels match the color).
    """
    # Load the image
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure the image is in RGB format

    # Initialize a dictionary to count the occurrences of each color in the palette
    color_counts = {color_name: 0 for _, color_name in PALETTE}

    # Iterate over each pixel in the image
    for pixel in img.getdata():
        # Compare the current pixel with each color in the palette
        for color_rgb, color_name in PALETTE:
            if pixel == tuple(color_rgb):
                color_counts[color_name] += 1

    # Filter the colors that appear at least 5 times
    detected_colors = [color_name for color_name, count in color_counts.items() if count >= 20]

    if 'Black' in detected_colors: # Remove black (invalid) from detected colors
        detected_colors.remove('Black')

    return detected_colors


def visualize_img_text_output(img_path_list, text_list, output_path=None):
    """
    To help visualize image and text at the same time
    """
    # Load the images
    images = [Image.open(img_path) for img_path in img_path_list]

    # Create a figure to display the images and annotations
    fig, axs = plt.subplots(1, len(images), figsize=(5*len(images), 15))

    # Display the images
    for i, (ax, img) in enumerate(zip(axs, images)):
        ax.imshow(img)
        ax.axis('off')
    plt.subplots_adjust(bottom=0.6)

    # Add the text annotations below the images
    for i, line in enumerate(text_list):
        plt.figtext(0.1, 0.6 - i*0.04, line, ha="left", va="top", fontsize=12)

    # Save the figure
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

