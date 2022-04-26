import csv
from pathlib import Path
from pprint import pprint
import os
import shutil


reddit_posts_csv_location = "/Users/cyrilvallez/Desktop/archive/reddit_posts.csv"

image_store = "/Users/cyrilvallez/Desktop/archive/images/kaggle_images"

reference_store = "/Users/cyrilvallez/Desktop/Templates"


def get_ref_memes():
    """
    Get the first occurences of a meme template in Kaggle dataset and registers it as a

    """

    template_to_first_occurence = {}

    with open(reddit_posts_csv_location, 'r', encoding='utf-8') as source_csv:
        reader = csv.reader(source_csv)
        next(reader)
        for i, line in enumerate(reader):
            if line[1] not in template_to_first_occurence.keys():
                template_to_first_occurence[line[1]] = line[0]

    stem_2_fname = {}

    for dirpath, dirnames, files in os.walk(image_store):
        for fname in files:
            stem_2_fname[Path(fname).stem] = fname

    Path(reference_store).mkdir(parents=True, exist_ok=True)

    for template, occurence in template_to_first_occurence.items():
        source_path = Path(image_store).joinpath(stem_2_fname[occurence])
        corrected_template_name = template.replace('_', '-')
        destination_path = Path(reference_store).joinpath(template + Path(stem_2_fname[
                                                                              occurence]).suffix)
        print("%s > %s" % (source_path, destination_path))
        # shutil.copyfile(source_path, destination_path)

    with open(reddit_posts_csv_location, 'r', encoding='utf-8') as source_csv:
        reader = csv.reader(source_csv)
        next(reader)
        for i, line in enumerate(reader):
            template = line[1]
            corrected_template_name = template.replace('_', '-')
            fname_stem = line[0]
            print(fname_stem)
            if fname_stem not in stem_2_fname.keys():
                print('stem not found')
                continue
            if '?' in fname_stem:
                fname_stem = fname_stem.replace('?', '')
            corrected_fname_stem = fname_stem.replace('_', '-')
            base_path = Path(image_store).joinpath(stem_2_fname[fname_stem])
            rename_path = Path(corrected_template_name + '_' + corrected_fname_stem +
                               base_path.suffix)
            rename_path = Path(image_store).joinpath(rename_path)
            print("%s > %s" % (base_path, rename_path))

            if base_path.exists():
                base_path.rename(rename_path)
            else:
                print(str(base_path) + 'not found on disk')

    # pprint(stem_2_fname)
    # pprint(template_to_first_occurence)


def extract_templates():

    met_memes = set()

    for _, _, fnames in os.walk(image_store):

        for fname in fnames:
            meme_ref = fname.split('_')[0]
            if meme_ref not in met_memes:
                met_memes.add(meme_ref)
                shutil.copy(Path(image_store).joinpath(fname),
                            Path(reference_store).joinpath(meme_ref + Path(fname).suffix))


if __name__ == "__main__":
    # get_ref_memes()
    extract_templates()