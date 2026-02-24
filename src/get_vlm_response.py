import os
import sys
from PIL import Image

from utils.vlm_utils import (
    encode_image, create_payload_gpt4, get_response, create_content_list, formulate_input,
    query_model_text_only,
    parse_lm_output
)

################
# Prompt
################
def end_to_end_matching_prompt(obj, cluster_color_str):
    system_prompt = [
        "Given (a) the category of an object, (b) a reference image of the original object,",
        "(c) an image visualizing clustering of the object into regions, each indicated by a distinct color,",
        "and (d) a list of used colors, your task is to:\n",
        "1. Identify the main functional regions of the object that agents would interact with during use.",
        "Focus on regions that have clear interactive purposes and practical functions. ",
        "For each identified functional region, provide a mix of action-oriented descriptions that capture how agents would interact with this region.\n",
        "2. Match each set of action descriptions to the most appropriate colored region in image (c).",
        "Carefully examine the colored regions in image (c) to ensure you match descriptions to the correct colors.\n",
        "Important rules to follow:\n",
        "- If you cannot identify the specified object category in the images, respond with only: \"ANSWER: CANNOT IDENTIFY TARGET OBJECT\"",
        "- Focus ONLY on direct agent-object interactions (e.g., 'grip', 'pull', 'where to hold') or agent-multi-object interactions (e.g., 'where to cut food with').",
        "- Do NOT include descriptions that merely state object functionality without agent action. Examples of what to avoid: 'swings open' for a cabinet door, 'stores items' for a drawer. Instead use: 'pull to open' for a cabinet door, 'place items in' for a drawer.",
        "- Distinct Functional Parts Rule: Carefully identify when a functionally distinct part (like a knife tip, scissor point, or button) has been separated into its own cluster. Each distinct functional part should receive its own specific action descriptions relating to its unique purpose, even if it's part of a larger component (like a blade).",
        "- Region Granularity Rule: If different functional parts that serve SIMILAR purposes are grouped into the same cluster, describe the main functionality for the entire region.",
        "- Multi-Region Functionality Rule: If a functional part spans multiple clusters, prioritize the largest/main region for that functionality.",
        "- Only propose descriptions for regions that have common, practical actions associated with them.",
        "- Do NOT force descriptions for every colored region if some lack clear interactive purposes.",
        "- Avoid mentioning body parts in your descriptions when possible.",
        "Do NOT use single quotes (') for any elements. Use double quotes (\") only.",
        "Pay careful attention to the output format instructions to ensure your response is correctly formatted and parseable."
    ]

    user_prompt = [
        f"The first image is the original image of the object, and the second image shows the clustering of regions in colors.\n",
        f"This is the color list: {cluster_color_str}, and this is the object category: {obj}.\n ",
        "Identify the main functional regions in these images and provide action-oriented descriptions for each region.\n",
        "If you cannot identify the specified object category in the images, respond with only: \"ANSWER: CANNOT IDENTIFY TARGET OBJECT\"\n",
        "First, carefully examine both images to match the colors in the clustered image with the corresponding regions in the original image.\n",
        "Make sure to identify all functionally distinct parts that have been given their own color cluster. For example, if a knife tip has its own color separate from the main blade, give it specific actions like 'pierce', 'puncture', or 'poke' that relate to its distinct functionality.\n",
        "For each functional region you identify, generate 4-5 descriptions including:\n",
        "- 1-2 simple action verbs (e.g., 'grip', 'cut', 'pour', 'pierce')",
        "- 1-2 action phrases (e.g., 'holding to stabilize', 'pressing to operate', 'poking to test')",
        "- 1-2 natural language descriptions (e.g., 'the region to grasp when cutting', 'where to apply pressure', 'the point to pierce packaging')\n",
        "Examples:\n",
        "For a knife blade: ['cut', 'slice', 'the edge for cutting food', 'where to position against materials for cutting']\n",
        "For a knife tip (if separate): ['pierce', 'puncture', 'the point for making initial cuts', 'where to poke to open packages']\n",
        "For a cup handle: ['grip', 'hold', 'the region to grasp when drinking', 'where to hold to avoid hot surfaces']\n",
        "For a drawer handle: ['pull', 'grasp', 'the part to pull to open drawer', 'where to grip to access contents']\n",
        "Remember:\n",
        "- When a functionally distinct part has its own color cluster (like a knife tip), give it specific actions related to its unique purpose",
        "- When parts with similar functions are in the same cluster, focus on the main functionality",
        "- When a functional part spans multiple clusters, focus on the largest/main region",
        "- Only describe regions with clear interactive purposes (ignore purely decorative elements)",
        "- Focus on actions an agent would perform (e.g., 'place items in'), not passive object functions (e.g., 'stores items')\n",
        "Output format: You MUST follow this exact format. Start with the word \"ANSWER: \", followed by a dict, where each key-value pair contains a list of action descriptions for the same region as the key, and the corresponding color as the value. The keys must be formatted as JSON-compatible strings (with backslashes before internal quotes): ",
        "ANSWER: {\"[\\\"grip\\\", \\\"hold\\\", \\\"the region to grasp when cutting\\\", \\\"where to hold firmly\\\"]\" : \"Red\", ",
        "\"[\\\"cut\\\", \\\"slice\\\", \\\"the edge for cutting food\\\", \\\"where to position against materials\\\"]\" : \"Blue\", ",
        "\"[\\\"pierce\\\", \\\"puncture\\\", \\\"the point for making initial cuts\\\", \\\"where to poke to open packages\\\"]\" : \"Green\"}\n",
        "The content after \"ANSWER: \" must be parseable with Python's ast.literal_eval(). ",
        "All elements in the lists should be enclosed by double quotes with proper escape characters (\\). ",
        "The color should only be one color, with the first letter capitalized and the rest in lowercase.",
        "Double-check that your output is properly formatted with correct JSON syntax and escaping of quotes before submitting."
    ]
    return system_prompt, user_prompt


################
# Main
################
def get_end_to_end_matching(obj, query_path, query_prefix, use_vlm='claude'):
    """
    Get the color matching for the region description.
    """

    # formulate input for query images
    encoded_imgs = []
    color_names = {}
    
    for prefix in query_prefix:
        encoded_imgs_, color_names_ = formulate_input(query_path, prefix)
        encoded_imgs.extend(encoded_imgs_)
        color_names.update(color_names_)

    # create prompt
    propose_name = f"{query_prefix[0]}_proposal"
    original_name = f"{query_prefix[0]}_original"
    print("using colors",color_names[propose_name])
    system_prompt, user_prompt = end_to_end_matching_prompt(obj, color_names[propose_name])

    if use_vlm == 'gpt':
        system_content = create_content_list(["".join(system_prompt)], None)
        user_content = create_content_list(["".join(user_prompt)], encoded_imgs)

        # query model
        payload = create_payload_gpt4([{"role":"system", "content":system_content}, {"role":"user", "content":user_content}])
        response = get_response(payload)
        text = response.choices[0].message.content

    else:
        raise NotImplementedError(f"Invalid VLM model: {use_vlm}")

    # parse output
    region_matching = parse_lm_output(text, parse_lst=False, parse_dict = True)

    return region_matching