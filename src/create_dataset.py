from concurrent.futures import ThreadPoolExecutor
import os
from PIL import Image
import requests
import threading

import cv2
import pandas as pd
from tqdm import tqdm


def contains_word(sentence, target_words):
    for token in sentence.split():
        if token in target_words:
            return True
    return False

def get_img_from_url(url):
    try:
        return Image.open(requests.get(url, stream=True, timeout=8).raw)
    except Exception as e:
        print('ERROR!' + e)
        print('Couldn\'t get img from url "{}" because of `{}`, skipping.'.format(url, type(e)))

def format_img(img, target_size=256):
    width, height = img.size
    smaller_dim = min(width, height)
    scale_factor = target_size / smaller_dim
    width = int(scale_factor * width)
    height = int(scale_factor * height)
    img = img.resize((width, height))

    half_size = target_size // 2
    left = (width // 2) - half_size
    right = (width // 2) + half_size
    top = (height // 2) - half_size
    bottom = (height // 2) + half_size
    img = img.crop((left, top, right, bottom))
    
    return img

def process_and_save_img(url, name, output_dir):
    img = get_img_from_url(url)
    img = format_img(img)
    
    save_path = os.path.join(output_dir, name)
    img.save('{}.png'.format(save_path), 'PNG')

def download_dataset(df, output_dir, n_threads=16):
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            name = str(row.name)
            url = row['url']
            executor.submit(process_and_save_img, url, name, output_dir)
    
def e2e_download_dataset(captions_path, output_dir, categories_path=None, n_threads=16):
    df = pd.read_csv(captions_path, delimiter='\t', header=None, names=['caption', 'url'])
    df = df.sample(frac=1)
    
    if categories_path:
        with open(categories_path, 'r') as f:
            categories = f.readlines()
        categories = set([c.strip().lower() for c in categories if ' ' not in c.strip()])

        category_entries = df.apply(lambda row: contains_word(row['caption'], categories), axis=1)
        df = df[category_entries]
    
    download_dataset(df, output_dir, n_threads=n_threads)


if __name__ == '__main__':
    data_dir = '../data/'
    img_dir = os.path.join(data_dir, 'google_captions/imgs/')
    data_fp = os.path.join(data_dir, 'google_captions/gcc_train_data.tsv')
    animal_names_fp = os.path.join(data_dir, 'animal_names.txt')

    df = pd.read_csv(data_fp, delimiter='\t', header=None, names=['caption', 'url'])
    df = df.sample(frac=1)
    df.head()

    with open(animal_names_fp, 'r') as f:
        animal_names = f.readlines()
    animal_names = set([an.strip().lower() for an in animal_names if ' ' not in an.strip()])
    print(len(animal_names), 'animals')

    animal_entries = df.apply(lambda row: contains_word(row['caption'], animal_names), axis=1)
    animal_df = df[animal_entries]
    print(len(animal_df))

    download_dataset(animal_df, img_dir)