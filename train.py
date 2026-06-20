import torch
import torch.nn as nn
from torch.nn import functional as F

# =========================
# LOAD CORPUS
# =========================

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Jumlah karakter:", len(text))

# =========================
# VOCABULARY
# =========================

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Jumlah vocab:", vocab_size)

# karakter ke angka
stoi = { ch:i for i,ch in enumerate(chars) }

# angka ke karakter
itos = { i:ch for i,ch in enumerate(chars) }

# encode text -> angka
encode = lambda s: [stoi[c] for c in s]

# decode angka -> text
decode = lambda l: ''.join([itos[i] for i in l])

# =========================
# TOKENIZATION
# =========================

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

# =========================
# TRAIN VALID SPLIT
# =========================

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Train data:", len(train_data))
print("Validation data:", len(val_data))

# =========================
# BATCH CONFIG
# =========================

batch_size = 16
block_size = 64

def get_batch(split):

    data = train_data if split == 'train' else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i+block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i+1:i+block_size+1]
        for i in ix
    ])

    return x, y

# =========================
# MODEL GPT
# =========================

class TinyGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            64
        )

        # linear output
        self.linear = nn.Linear(
            64,
            vocab_size
        )

    def forward(self, idx, targets=None):

        x = self.embedding(idx)

        logits = self.linear(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B*T, C)

            targets = targets.view(B*T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self(idx)

            logits = logits[:, -1, :]

            probs = F.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx

# =========================
# INIT MODEL
# =========================

model = TinyGPT()

print(model)

# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)

# =========================
# TRAINING
# =========================

max_iters = 500

for step in range(max_iters):

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()

    if step % 50 == 0:

        print(
            f"step {step} | loss {loss.item():.4f}"
        )

# =========================
# GENERATE TEXT
# =========================

print("\n======================")
print("HASIL GENERATE TEXT")
print("======================\n")

context = torch.zeros(
    (1,1),
    dtype=torch.long
)

hasil = model.generate(
    context,
    max_new_tokens=500
)[0].tolist()

print(decode(hasil))

# =========================
# SAVE OUTPUT
# =========================

with open(
    "hasil_output.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(decode(hasil))

print("\nHasil generate disimpan ke hasil_output.txt")
