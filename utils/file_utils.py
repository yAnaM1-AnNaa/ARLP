import h5py
import matplotlib.pyplot as plt
import yaml
import textwrap
import pandas as pd
import logging
import sqlite3
import os
import numpy as np

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

class SolveFolder():
  def list_image_paths_recursive(img_dir, image_extensions):
      image_paths = []
      for root, _, files in os.walk(img_dir):
          for name in sorted(files):
              ext = os.path.splitext(name)[1].lower()
              if ext in image_extensions:
                  image_paths.append(os.path.join(root, name))
      image_paths.sort()
      return image_paths
  
  def save_local_result(image_path, img, heatmap, text_query, output_dir):
      os.makedirs(output_dir, exist_ok=True)

      base_name = os.path.splitext(os.path.basename(image_path))[0]
      heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.npy")
      overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")

      np.save(heatmap_path, heatmap)

      plt.figure(figsize=(8, 4))

      plt.subplot(1, 2, 1)
      plt.imshow(img)
      plt.title("Input")
      plt.axis("off")

      plt.subplot(1, 2, 2)
      plt.imshow(img)
      mask_result = np.ma.masked_where(heatmap < 0.001, heatmap)
      plt.imshow(mask_result, cmap="jet", alpha=0.5, vmin=0, vmax=1)
      plt.title(text_query)
      plt.axis("off")

      plt.tight_layout()
      plt.savefig(overlay_path)
      plt.close()

      return {
          "heatmap_npy": heatmap_path,
          "overlay_png": overlay_path,
      }


class ResearchSql():
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row # row_factory 使查询结果可以像字典一样访问
        self.cursor = self.conn.cursor()
        self._init_table_struture()
        # self._init_table_content(base_dir)

    def _init_table_struture(self):
        query = '''
        CREATE TABLE IF NOT EXISTS pipeline_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT,
            instance_name TEXT,
            frame_idx TEXT,
            clustered_img_path TEXT,
            status TEXT,
            vlm_response TEXT,
            error_msg TEXT,
            UNIQUE(category_name, instance_name, frame_idx)
        )
        '''
        self.cursor.execute(query)
        self.conn.commit()

    # def _init_table_content(self, base_dir):
    #     h5_paths = []
    #     for name in os.listdir(base_dir):
    #         path = os.path.join(base_dir, name)
    #         if not os.path.isfile(path):
    #             continue
    #         if not name.endswith('.h5'):
    #             continue
    #         category_name = name.split('.')[0]
    #         with h5py.File(path, 'r') as h5_file:
    #             instance_keys = list(h5_file.keys()) # chair1, chair2...
    #             for instance_name in instance_keys:
    #                 self.add_data(category_name, instance_name, '', '', '', '', '')

    def add_data(self, category_name, instance_name, frame_idx, clustered_img_path, status, vlm_response, error_msg):
        query = '''
        INSERT INTO pipeline_info (category_name, instance_name, frame_idx, clustered_img_path, status, vlm_response, error_msg)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(category_name, instance_name, frame_idx) 
        DO UPDATE SET 
            clustered_img_path = excluded.clustered_img_path,
            status = excluded.status,
            error_msg = excluded.error_msg
        '''
        self.cursor.execute(query, (category_name, instance_name, frame_idx, clustered_img_path, status, vlm_response, error_msg))
        self.conn.commit()

    def update_content(self, category_name, instance_name, frame_idx,  
                       clustered_img_path=None, status=None, vlm_response=None, error_msg=None):
        update_fields = []
        update_values = []

        if frame_idx is not None:
            update_fields.append('frame_idx = ?')
            update_values.append(frame_idx)

        if clustered_img_path is not None:
            update_fields.append('clustered_img_path = ?')
            update_values.append(clustered_img_path)

        if status is not None:
            update_fields.append('status = ?')
            update_values.append(status)

        if vlm_response is not None:
            update_fields.append('vlm_response = ?')
            update_values.append(vlm_response)

        if error_msg is not None:
            update_fields.append('error_msg = ?')
            update_values.append(error_msg)

        if not update_fields:
            raise ValueError('No content provided to update.')

        query = f'''
        UPDATE pipeline_info
        SET {', '.join(update_fields)}
        WHERE category_name = ? AND instance_name = ? AND frame_idx = ?
        '''

        update_values.extend([category_name, instance_name, frame_idx])
        self.cursor.execute(query, tuple(update_values))
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

    def delete_data(self, category_name, instance_name, frame_idx=''):
        query = '''
        DELETE FROM pipeline_info
        WHERE category_name = ? AND instance_name = ? AND frame_idx = ?
        '''
        self.cursor.execute(query, (category_name, instance_name, frame_idx))
        self.conn.commit()
        
    def close(self):
        self.conn.close()


class store_logs():
    def __init__(self, log_name = 'pipeline_logs', log_file = 'dataset/pipeline_logs.txt'):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        self.logger.addHandler(file_handler)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
    @property
    def handlers(self):
        return self.logger.handlers

    def record(self, content, *args, level='info'):
        log_fn = self.logger.info
        if level.lower() == "error":
            log_fn = self.logger.error
        elif level.lower() == "warning":
            log_fn = self.logger.warning

        if args:
            log_fn(content, *args)
        else:
            log_fn(content)

    def info(self, content, *args):
        self.logger.info(content, *args)

    def warning(self, content, *args):
        self.logger.warning(content, *args)

    def error(self, content, *args):
        self.logger.error(content, *args)

    def exception(self, content, *args):
        self.logger.exception(content, *args)