from pathlib import Path
from marcussen.dataset import MarcussenDataset
from marcussen.compare import run_within_group

# Load dataset
dataset_root = "Marcussen 1945-1975_FLAC_Dataset"
dataset = MarcussenDataset(dataset_root)

# Ensure scanning happens
all_items = dataset.flat_items_list()  # safely populates _items

# Filter items by metadata
filtered_items = [
    item for item in all_items
    if getattr(item, "meta", None) is not None
    and item.meta.get("family") == "Principals"
    and item.meta.get("registration_raw") == "P8"
    and item.meta.get("division") == "upper_division"
    and item.meta.get("mic_location") == "close"
    and item.meta.get("normalisation") == "yes"
]

# Wrap filtered items for run_within_group
from collections import defaultdict

class FilteredDatasetWrapper:
    def __init__(self, items):
        self._filtered_items = items

    def class_groups(self):
        grouped = defaultdict(list)

        for item in self._filtered_items:
            pitch = item.meta.get("pitch", "unknown")
            grouped[pitch].append(item)

        return dict(grouped)

# Create wrapper
filtered_dataset = FilteredDatasetWrapper(filtered_items)

# Output CSV
output_csv = Path("outputs/pairs_within_group_filtered_p8.csv")
output_csv.parent.mkdir(exist_ok=True, parents=True)

# Run within-group comparisons
result = run_within_group(
    dataset=filtered_dataset,
    out_csv_path=output_csv,
    metric="fad_clap_audio",
    max_pairs=None
)

print("Comparison done!")
print(result)