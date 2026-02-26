import h5py
import matplotlib.pyplot as plt
import yaml
import textwrap

def save_response(cot_resoponse, logger):
    num_question = int(len(cot_resoponse))
    q = 0
    while True:
        raw_sentence = str(cot_resoponse[q])
        wrapped_sentence = textwrap.fill(
            raw_sentence,
            width=110,
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent='  '
        )
        if q % 2 == 0:
            logger.info(f'Question {int(1 + q/2)}:\n{wrapped_sentence}')
        elif q % 2 == 1:
            logger.info(f'Response:\n{wrapped_sentence}')
        q += 1
        if q == num_question:
            break

def store_or_update_dataset(h5_group, dataset_key, data, compression=None):
    """
    Store or update a dataset within an HDF5 group.
    
    If the dataset_key already exists, it is deleted first (to handle
    shape or type changes). Then a new dataset is created.
    
    If data is a list of Python strings, it is stored as a variable-length
    string array. Otherwise, it is stored as is.
    """
    import h5py

    # Remove existing dataset if present
    if dataset_key in h5_group:
        del h5_group[dataset_key]

    # If data is a list of strings, store as variable-length strings
    if (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], str)
    ):
        dtype = h5py.special_dtype(vlen=str)
        dset = h5_group.create_dataset(
            dataset_key,
            shape=(len(data),),
            dtype=dtype,
            compression=compression
        )
        dset[:] = data
    else:
        # For numeric arrays, just store them directly
        h5_group.create_dataset(
            dataset_key,
            data=data,
            compression=compression
        )

def save_image(image, path):
    """Save image using matplotlib."""
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)