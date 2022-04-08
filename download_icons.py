from data_preprocessing import DataPreprocessing
from flaticon_api import FlaticonApi

# ---- Get the list of tags ---- #
tags_filepath = "tags.txt"

print(f"--- Loading tags from {tags_filepath} ...")
with open(tags_filepath) as tags_file:
    tags = [line.rstrip() for line in tags_file]
tags = tags[:5]
tags = [tag.lower() for tag in tags]
onehot_tags = DataPreprocessing.map_tags_into_onehot(tags)
print(f"--- {len(tags)} tags loaded ---")
# ------------------------------- #

# ------- Download png from the list of tags ---------- #
api = FlaticonApi()
limit_per_tag = 1000
icon_size = "32"

print("--- Download icons links and tags ---")
all_ic_urls = []
all_ic_tags = []
for tag in tags:
    data, metadata = api.get_black_icons(tag, limit_per_tag)
    ic_urls = [d["images"][icon_size] for d in data]
    ic_tags = [d["tags"].split(',') for d in data]
    # The tag searched may not appear in the tags, so we add it
    for ft in ic_tags:
        if tag not in ft:
            ft.append(tag)
    all_ic_urls = [*all_ic_urls, *ic_urls]
    all_ic_tags = [*all_ic_tags, *ic_tags]
print(f"--- Links and tags downloaded. Total : {len(all_ic_urls)} ---")
# ----------------------------------------------------- #

# ---- Save ---- #
# Save tags
with open('ic_tags_5.txt', 'w') as f:
    for item in all_ic_tags:
        f.write("%s\n" % item)

# Save links
with open('links_5.txt', 'w') as f:
    for item in all_ic_urls:
        f.write("%s\n" % item)
# -------------- #

# ---------- Download icons from links ------- #

print("--- Downloading icons ---")
dp = DataPreprocessing(data_path='./data_5', filecount=0)
dp.download_images(all_ic_urls)
print("--- Icons downloaded ---")

# ------------------------------------------ #


