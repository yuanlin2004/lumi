import pickle

with open('sampler_history.pkl', 'rb') as f:
    history = pickle.load(f)

#print(history)

# history is a list of [(n, indices, values, picked, logits), str, ...]
for item in history:
    if isinstance(item, str):
        print(f"{item[:20]:20} - Cut @ {n:5}  Highest @ {indices:8} with {values:.2f}  Picked {picked:8} with {logits:.2f}")
    elif isinstance(item, tuple):
        n, indices, values, picked, logits = item