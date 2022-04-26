"""
author: Andrei Kucharavy (@chiffa)

"""
from imgur_scraper.imgur_scraper import get_viral_posts_from
import requests
import csv
import os
from pathlib import Path
from pprint import pprint
# from local_secrets import client_id
# import json
import datetime
import time
from tqdm import tqdm


requests_header = {"Authorization": "Client-ID fb6b52033c2c4d9"}

csv_storage = "metadata_buffer"
Path(csv_storage).mkdir(parents=True, exist_ok=True)

image_storage = "image_data"
Path(image_storage).mkdir(parents=True, exist_ok=True)


def load_metadata_and_url(start_date: str, end_date: str, provide_details: bool = False):
    """
    Basically clone of the original repo's command line parser.
    Just goes and loads the metadata and album urls from Imgur most viral

    """

    try:
        file_name = os.path.join(
            csv_storage, f"{start_date}_to_{end_date}_imgur_data.csv")



        with open(file_name, "x", newline="", encoding="utf-8") as csvfile:
            if provide_details:
                fieldnames = ["title", "url", "points", "tags", "type", "views", "date", "username",
                              "comment_count", "downs", "ups", "points", "score", "timestamp",
                              "views", "favorite_count", "hot_datetime", "nsfw", "platform",
                              "virality"]

            else:
                fieldnames = ["title", "url", "points",
                              "tags", "type", "views", "date"]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(get_viral_posts_from(
                start_date, end_date))

        print(f"CSV saved in {os.path.abspath(file_name)}")

    except FileExistsError as f:
        print(f)

    except ValueError as v:
        print(v)


def load_image_from_url(album_directory: Path, image_url: str) -> None:
    """
    Loads image from URL and resizes it to a large thumbnail

    """
    thumbnail_url = image_url.split('.')

    if '?' in thumbnail_url[-1]:
        thumbnail_url[-1] = thumbnail_url[-1].split('?')[0]

    image_fname = thumbnail_url[-2].split('/')[-1] + '.' + thumbnail_url[-1]
    save_file = album_directory.joinpath(image_fname)

    thumbnail_url[-2] = thumbnail_url[-2] + 'l'

    thumbnail_url = '.'.join(thumbnail_url)

    # print(thumbnail_url)
    # print(save_file)

    img_data = requests.get(thumbnail_url).content

    with open(save_file, 'wb') as handler:
        handler.write(img_data)

    time.sleep(1)  # courtesy rate limiter


def load_images_from_gallery(album_hash: str = "CwK7OKz") -> None:

    album_path = Path(image_storage).joinpath(album_hash)
    # print(album_path)

    try:
        album_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print('\tskipping since exists %s' % album_hash)
        return  # this performs an auto-skip allowing to resume downloads without using up rate
                #   limiting

    album_url = "https://api.imgur.com/3/album/%s/images" % album_hash

    # print(album_url)
    # print(requests_header)

    resp = requests.get(album_url, headers=requests_header)
    # print(resp.status_code)
    # print(resp.headers)
    try:
        print("\t%s: %d; client limit: %s/%s; user limit: %s/%s; user reset %s" %
              (album_hash, resp.status_code,
               resp.headers['X-RateLimit-ClientRemaining'], resp.headers['X-RateLimit-ClientLimit'],
               resp.headers['X-RateLimit-UserRemaining'], resp.headers['X-RateLimit-UserLimit'],
               datetime.datetime.fromtimestamp(int(resp.headers['X-RateLimit-UserReset'])).isoformat()))

        if int(resp.headers['X-RateLimit-ClientRemaining']) < 5 or \
            int(resp.headers['X-RateLimit-UserRemaining']) < 5:
            raise Exception('Rate limit reached, backing off')

    except KeyError as e:
        print(e)
        time.sleep(0.5)  # time for print to finish writing other things
        print(album_hash)
        print(resp.status_code)
        pprint(resp.headers)
        pprint(resp.text)
        input('press any key to continue')


    # print(resp.text)
    # pprint(resp.json())

    for image_dict in tqdm(resp.json()['data']):
        if image_dict['is_ad']:
            print('ad, skipping')
            continue

        if image_dict['animated']:
            print('is a gif/mp4, skipping')
            continue

        if image_dict['nsfw']:
            print('is nsfw, skipping')
            continue

        url = image_dict["link"]

        load_image_from_url(album_path, url)


def load_images_from_a_crawl(album_crawl_dataset: Path):
    """
    Takes the results of the Imgur albums crawl and traverses it

    """
    print('traversing crawl %s' % album_crawl_dataset)

    with open(album_crawl_dataset, 'r', encoding='utf-8') as source_file:
        reader = csv.reader(source_file)
        next(reader)

        for data_line in reader:
            album_hash = data_line[1].split('/')[-1]
            # print(album_hash)

            if not data_line[4] == 'album':
                print('skipping %s, since not an album' % album_hash)
                continue

            load_images_from_gallery(album_hash=album_hash)


if __name__ == "__main__":
    load_metadata_and_url("2019-12-15", "2020-01-03", False)
    active_crawl_dataset = Path("/Users/cyrilvallez/Desktop/Project/data_retrieval/metadata_buffer/2019-12-15_to_2020-01-03_imgur_data.csv")
    load_images_from_a_crawl(active_crawl_dataset)
    # load_images_from_gallery()

