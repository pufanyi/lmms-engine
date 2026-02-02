import os
from typing import Dict

import torch
from PIL import Image

from lmms_engine.datasets.collator import VisionCollator
from lmms_engine.datasets.iterable.multimodal_iterable_dataset import MultiModalIterableDataset
from lmms_engine.mapping_func import register_dataset


@register_dataset("qwen3_vl_iterable")
class Qwen3VLIterableDataset(MultiModalIterableDataset):
    """
    Dataset for Qwen3-VL training with conversations format.

    Expected data format (JSONL):
    {
        "id": "sample_id",
        "conversations": [
            {"from": "human", "value": "<image><image> Question text..."},
            {"from": "gpt", "value": "Answer text..."}
        ],
        "image": ["path/to/image1.jpg", "path/to/image2.jpg"]
    }
    """

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        conversations = data["conversations"]
        image_paths = data.get("image", [])

        # Build HF-style messages with image placeholders
        hf_messages = []
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            text = conv["value"]

            # Parse <image> tags and build content list
            content = []
            parts = text.split("<image>")

            for i, part in enumerate(parts):
                # Add image placeholder before each part (except the first)
                if i > 0:
                    content.append({"type": "image"})
                # Add text if non-empty
                if part:
                    content.append({"type": "text", "text": part})

            # If content is empty, add empty text
            if not content:
                content.append({"type": "text", "text": ""})

            hf_messages.append({"role": role, "content": content})

        # Load images
        images = None
        if image_paths:
            images = []
            for img_path in image_paths:
                full_path = os.path.join(data_folder, img_path) if data_folder else img_path
                images.append(Image.open(full_path))

        return self.processor.process(images=images, hf_messages=hf_messages)

    def get_collator(self):
        return VisionCollator(self.processor)
