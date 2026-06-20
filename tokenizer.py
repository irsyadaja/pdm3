import sentencepiece as spm

# =========================
# LOAD CORPUS
# =========================

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Jumlah kata:", len(text.split()))

# =========================
# CHARACTER TOKENIZER
# =========================

print("\n========================")
print("CHARACTER TOKENIZER")
print("========================")

chars = sorted(list(set(text)))

print("Jumlah karakter unik:", len(chars))

sample_text = "Pariwisata Indonesia sangat indah"

char_tokens = list(sample_text)

print("\nText:")
print(sample_text)

print("\nCharacter Tokens:")
print(char_tokens)

# =========================
# BPE TOKENIZER
# =========================

print("\n========================")
print("BPE TOKENIZER")
print("========================")

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer_bpe',
    vocab_size=100,
    model_type='bpe'
)

sp_bpe = spm.SentencePieceProcessor()

sp_bpe.load("tokenizer_bpe.model")

bpe_tokens = sp_bpe.encode(
    sample_text,
    out_type=str
)

print("\nText:")
print(sample_text)

print("\nBPE Tokens:")
print(bpe_tokens)

# =========================
# UNIGRAM TOKENIZER
# =========================

print("\n========================")
print("UNIGRAM TOKENIZER")
print("========================")

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer_unigram',
    vocab_size=100,
    model_type='unigram'
)

sp_uni = spm.SentencePieceProcessor()

sp_uni.load("tokenizer_unigram.model")

uni_tokens = sp_uni.encode(
    sample_text,
    out_type=str
)

print("\nText:")
print(sample_text)

print("\nUnigram Tokens:")
print(uni_tokens)

# =========================
# SAVE RESULT
# =========================

with open(
    "tokenizer_result.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("=== CHARACTER TOKENIZER ===\n")
    f.write(str(char_tokens))

    f.write("\n\n=== BPE TOKENIZER ===\n")
    f.write(str(bpe_tokens))

    f.write("\n\n=== UNIGRAM TOKENIZER ===\n")
    f.write(str(uni_tokens))

print("\nHasil tokenizer disimpan ke tokenizer_result.txt")