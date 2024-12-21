import os
import json
from tqdm import tqdm


def beautify_large_number(large_number: int) -> str:
    """Beautify large number by setting M, B, T, K suffixes."""
    if large_number >= 1_000_000_000_000:
        return f"{large_number / 1_000_000_000_000:.2f}T"
    if large_number >= 1_000_000_000:
        return f"{large_number / 1_000_000_000:.2f}B"
    if large_number >= 1_000_000:
        return f"{large_number / 1_000_000:.1f}M"
    if large_number >= 1_000:
        return f"{large_number // 1_000}K"
    
    return f"{large_number}"


root = "./hf_datasets/OpenMathInstruct2-ScaleQuest"
src_file = os.path.join(root, "data.jsonl")
tgt_dir = os.path.join(root, "formatted_data")

all_data = []
with open(src_file, "r") as f:
    for line in tqdm(f.readlines()):
        data = json.loads(line)
        new_data = {
            "instruction": data["problem"],
            "input": "",
            "output": data["generated_solution"],
        }
        all_data.append(new_data)

from random import shuffle
shuffle(all_data)

save_every_number_of_items = 100_000

accumulated_data = []
for i, data in enumerate(tqdm(all_data)):
    accumulated_data.append(data)
    if (i + 1) % save_every_number_of_items == 0:
        tgt_subdir = os.path.join(tgt_dir, f"train_{beautify_large_number(i + 1)}")
        os.makedirs(tgt_subdir, exist_ok=True)
        
        max_number_of_items_per_chunk = 100_000
        accumulated_data_chunks = [
            accumulated_data[i:i + max_number_of_items_per_chunk]
            for i in range(0, len(accumulated_data), max_number_of_items_per_chunk)
        ]
        for chunk_id, chunk in enumerate(accumulated_data_chunks):
            tgt_file = os.path.join(tgt_subdir, f"chunk{chunk_id}.json")
            with open(tgt_file, "w") as f:
                json.dump(chunk, f)