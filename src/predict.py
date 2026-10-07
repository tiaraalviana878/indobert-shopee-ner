"""
Prediksi atribut produk dari teks ulasan.
Jalankan: python src/predict.py "Sepatu Nike warna hitam ukuran 42 awet tapi sol cepat lepas"
"""
import sys

from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="models/indobert-ner",
    aggregation_strategy="simple",
    device="mps",  # ganti "cpu" jika error
)

text = " ".join(sys.argv[1:]) or "Sepatu Nike warna hitam ukuran 42 awet tapi sol cepat lepas, harga murah"
for e in ner(text):
    print(f"{e['entity_group']:<10} {e['word']:<20} {e['score']:.2f}")
