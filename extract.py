import gzip
import shutil

print("Extracting L2 dataset...")
with gzip.open('data/BTCUSDT_L2.csv.gz', 'rb') as f_in:
    with open('data/BTCUSDT_L2.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print("Extraction complete!")

print("First 5 lines:")
with open('data/BTCUSDT_L2.csv', 'r') as f:
    for _ in range(5):
        print(f.readline().strip())
