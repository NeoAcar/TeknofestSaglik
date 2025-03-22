import os
import pandas as pd
from PIL import Image
from torchvision.transforms import ToPILImage
from models.biomedclip_encoder import ImageEncoderWithMLP

input_csv = "data/data_paths/train_data.csv"
output_dir = "preprocessed/images"
output_csv = "preprocessed/train_preprocessed.csv"
os.makedirs(output_dir, exist_ok=True)

model = ImageEncoderWithMLP(
    config_path="configs/open_clip_config.json",
    checkpoint_path="checkpoints/open_clip_pytorch_model.bin"
)
preprocess_fn = model.get_preprocess()
to_pil = ToPILImage()

df = pd.read_csv(input_csv)
filepaths = df["filepath"].tolist()
labels = df["label"].tolist()

new_records = []
start_idx = 10000

for i, (path, label) in enumerate(zip(filepaths, labels)):
    image = Image.open(path).convert("RGB")
    image_tensor = preprocess_fn(image)
    out_image = to_pil(image_tensor)
    
    new_filename = f"{start_idx + i}.png"
    new_path = os.path.join(output_dir, new_filename)
    out_image.save(new_path)
    
    new_records.append({"filepath": new_path, "label": label})

pd.DataFrame(new_records).to_csv(output_csv, index=False)
print(f"Done. Saved {len(new_records)} images to {output_dir}")
print(f"CSV written to {output_csv}")
