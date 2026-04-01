import h5py
import matplotlib.pyplot as plt
import yaml
import textwrap
import pandas as pd
import logging
import sqlite3

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

class ResearchSql():
    def __init__(self, db_name = 'pipeline_info.db'):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row # row_factory 使查询结果可以像字典一样访问
        self.cursor = self.conn.cursor()
        self._init_table()

    def _init_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS pipeline_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_name TEXT,
            frame_idx TEXT,
            clustered_img_path TEXT,
            status TEXT,
            vlm_response TEXT,
            error_msg TEXT,
            UNIQUE(instance_name, frame_idx)
        )
        '''
        self.cursor.execute(query)
        self.conn.commit()

    def add_data(self, instance_name, frame_idx, clustered_img_path, status, vlm_response, error_msg):
        query = '''
        INSERT INTO pipeline_info (instance_name, frame_idx, clustered_img_path, status, vlm_response, error_msg)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(instance_name, frame_idx) 
        DO UPDATE SET 
            clustered_img_path = excluded.clustered_img_path,
            status = excluded.status,
            error_msg = excluded.error_msg
        '''
        self.cursor.execute(query, (instance_name, frame_idx, clustered_img_path, status, vlm_response, error_msg))
        self.conn.commit()

    def update_status(self, instance_name, frame_idx, status, error_msg=None):
        query = '''
        UPDATE pipeline_info
        SET status = ?, error_msg = ?
        WHERE instance_name = ?, frame_idx = ?
        '''
        self.cursor.execute(query, (status, error_msg, instance_name, frame_idx))
        self.conn.commit()

    def get_instance_data(self, instance_name):
        query = '''
        SELECT *
        FROM pipeline_info
        WHERE instance_name = ?
        ORDER BY frame_idx ASC
        '''
        self.cursor.execute(query, (instance_name,))
        rows = self.cursor.fetchall()
        result = [dict(row) for row in rows]
        return result

    def close(self):
        self.conn.close()


class store_logs():
    def __init__(self, log_name = 'pipeline_logs', log_file = 'pipeline_logs.txt'):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        self.logger.addHandler(file_handler)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)

    def record(self, content, level='info'):
        if level.lower() == "info":
            self.logger.info(content)
        elif level.lower() == "error":
            self.logger.error(content)
        elif level.lower() == "warning":
            self.logger.warning(content)