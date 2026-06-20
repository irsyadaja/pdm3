# TinyGPT Pariwisata Indonesia

Project sederhana implementasi tinyGPT menggunakan corpus bertema pariwisata Indonesia dengan beberapa pendekatan tokenisasi.

## Deskripsi Project

Project ini dibuat untuk mempelajari dasar dasar Large Language Model (LLM) sederhana menggunakan arsitektur tinyGPT berbasis PyTorch. Model dilatih menggunakan corpus bahasa Indonesia bertema pariwisata dengan jumlah lebih dari 2000 kata.

Selain melakukan training model, project ini juga membandingkan beberapa metode tokenisasi yaitu:

* Character Tokenization
* BPE (Byte Pair Encoding)
* Unigram Tokenization

---

# Dataset / Corpus

Corpus dibuat secara manual dengan topik pariwisata Indonesia.

Isi corpus mencakup:

* wisata alam
* wisata budaya
* wisata kuliner
* ekowisata
* teknologi pariwisata
* pemasaran digital wisata
* hotel dan transportasi
* pariwisata berkelanjutan

Total corpus:

* > 2000 kata

File:

```text
corpus.txt
```

---

# Struktur Project

```text
tinygpt-pariwisata/
│
├── corpus.txt
├── train.py
├── tokenizer.py
├── hasil_output.txt
├── tokenizer_result.txt
├── tokenizer_bpe.model
├── tokenizer_unigram.model
├── README.md
└── requirements.txt
```

---

# Install Library

```bash
pip install torch
pip install sentencepiece
```

atau:

```bash
pip install -r requirements.txt
```

---

# Training Model

Jalankan training model menggunakan:

```bash
python train.py
```

Training dilakukan menggunakan:

* PyTorch
* Character Level Tokenization
* AdamW Optimizer
* Cross Entropy Loss

Model akan:

1. membaca corpus
2. melakukan tokenisasi
3. training model
4. generate text otomatis

---

# Tokenizer

Project ini menggunakan beberapa pendekatan tokenizer:

## 1. Character Tokenizer

Text dipecah menjadi karakter per karakter.

Contoh:

```text
Pariwisata
```

menjadi:

```text
['P', 'a', 'r', 'i', 'w', 'i', 's', 'a', 't', 'a']
```

---

## 2. BPE Tokenizer

Menggunakan SentencePiece dengan metode Byte Pair Encoding.

Contoh hasil tokenisasi:

```text
['▁Pariwisata', '▁Indonesia', '▁sangat', '▁indah']
```

---

## 3. Unigram Tokenizer

Menggunakan SentencePiece dengan metode unigram.

Contoh hasil tokenisasi:

```text
['▁Pari', 'wisata', '▁Indonesia', '▁indah']
```

---

# Menjalankan Tokenizer

```bash
python tokenizer.py
```

Hasil tokenizer akan disimpan pada:

```text
tokenizer_result.txt
```

---

# Hasil Training

Selama proses training, nilai loss mengalami penurunan secara bertahap.

Contoh:

```text
step 0   | loss 4.12
step 50  | loss 3.21
step 100 | loss 2.75
step 200 | loss 2.10
```

Hal ini menunjukkan model mulai mempelajari pola teks dari corpus.

---

# Hasil Generate Text

Contoh hasil generate text:

```text
Pariwisata Indonesia memiliki banyak destinasi wisata alam dan budaya yang menarik wisatawan dari berbagai negara...
```

Hasil generate akan disimpan pada:

```text
hasil_output.txt
```

---

# Analisis

## Character Tokenization

Kelebihan:

* sederhana
* tidak ada unknown token

Kekurangan:

* sequence sangat panjang
* training lebih lambat

## BPE Tokenization

Kelebihan:

* efisien
* vocabulary lebih stabil

Kekurangan:

* perlu training tokenizer

## Unigram Tokenization

Kelebihan:

* fleksibel
* baik untuk subword

Kekurangan:

* hasil token terkadang terlalu kecil

---

# Kesimpulan

Model tinyGPT sederhana berhasil dilatih menggunakan corpus pariwisata bahasa Indonesia. Percobaan beberapa metode tokenisasi menunjukkan bahwa metode BPE menghasilkan tokenisasi yang lebih stabil dibanding character tokenizer dan unigram tokenizer.

Semakin besar corpus dan semakin lama training dilakukan, maka kualitas teks yang dihasilkan model akan semakin baik.

---

# Author

Nama: [ISI NAMA KAMU]

Mata Kuliah:
Big Data & Predictive Analytics

Semester:
4 Informatika
