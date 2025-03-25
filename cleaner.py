import os
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


input_base = "inme"
output_base = "inme"
folders = {
    "external_dataset/PNG": "external_clean"
}

# Temizleme fonksiyonu
def clean_image(paths):
    in_path, out_path = paths
    try:
        img = Image.open(in_path).convert("RGB")
        arr = np.array(img)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mask = ~((r == g) & (g == b))  # gri değilse
        arr[mask] = [0, 0, 0]
        Image.fromarray(arr).save(out_path)
    except Exception as e:
        print(f"Hata: {in_path} -> {e}")




if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Windows için güvenli başlatma

    for src, dst in folders.items():
        input_dir = os.path.join(input_base, src)
        output_dir = os.path.join(output_base, dst)
        os.makedirs(output_dir, exist_ok=True)

        jobs = []
        for fname in os.listdir(input_dir):
            if fname.lower().endswith(".png"):
                in_path = os.path.join(input_dir, fname)
                out_path = os.path.join(output_dir, fname)
                jobs.append((in_path, out_path))

        print(f"{src} klasörü işleniyor ({len(jobs)} dosya)...")

        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(clean_image, jobs), total=len(jobs)))
