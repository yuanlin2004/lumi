import pickle
import argparse

parser = argparse.ArgumentParser(description='Read history file')
parser.add_argument('history_file', type=str, help='History file to read')
args = parser.parse_args()

with open(args.history_file, 'rb') as f:
    history = pickle.load(f)

#print(history)

# history is a list of [(n, indices, values, picked, logits), str, ...]
for item in history:
    print(item)
    if isinstance(item, str):
        print(f"{item[:20]:20} - Cut @ {n:5}  Highest @ {indices:8} with {values:.2f}  Picked {picked:8} with {logits:.2f}")
    elif isinstance(item, tuple):
        n, indices, values, picked, logits = item