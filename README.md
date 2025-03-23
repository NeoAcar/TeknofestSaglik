# TeknofestSaglik

conda create -n biomedclip python=3.10 -y
conda activate biomedclip
pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib


for cuda remove replace torch with torch 2.5.1+cu"cudaversion" like 2.5.1.cu121 replace torchvision with 0.20.1cucu"cudaversion"

downnload data zip from the link in data.txt than extract

run download_bioclip.py to download model
run train.py for training 
run evaluate.py for evaluation
run plot_logs.py with desired csv path