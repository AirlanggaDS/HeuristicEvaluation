import pandas as pd

# Membaca dataset CSV
data = pd.read_csv('Form Survey KAI Access.csv')

# Mengonversi seluruh dataframe ke tipe data numerik
data = data.apply(pd.to_numeric, errors='coerce')

# Menggantikan nilai NaN dengan 0
data = data.fillna(0)

# Mengonversi seluruh dataframe ke tipe data integer
data = data.astype(int)

# Drop kolom 'Cap waktu'
data = data.drop(columns=['Cap waktu'])

# Mengubah nama kolom 'Nama atau inisial' menjadi 'ID'
data = data.rename(columns={'Nama atau inisial': 'ID'})

# Menambahkan kolom 'ID' sebagai indikator ID
data['ID'] = range(1, len(data) + 1)

# Menampilkan dataframe setelah modifikasi
print("Dataset")
print(data)

# Menghitung jumlah masing-masing kategori skor
instruments = data.drop(columns={'ID'})
jumlah_setuju = instruments.apply(lambda x: x.value_counts()).fillna(0).astype(int).transpose()

# Menampilkan hasil
print("\nJumlah masing-masing kategori skor:")
print(jumlah_setuju)

# Menghitung total keseluruhan untuk setiap nilai (1-6)
total_keseluruhan = jumlah_setuju.sum(axis=0)

# Menghitung proporsi keseluruhan untuk setiap nilai (1-6)
proporsi_keseluruhan = total_keseluruhan / total_keseluruhan.sum()

# Menampilkan hasil total keseluruhan
print("\nTotal keseluruhan untuk setiap nilai (1-6):")
print(total_keseluruhan)

# Menampilkan hasil proporsi keseluruhan
print("\nProporsi keseluruhan untuk setiap nilai (1-6):")
print(proporsi_keseluruhan)

from scipy.stats import t
import math
import numpy as np
from scipy.stats import pearsonr
from pingouin import reliability

# Menghitung jumlah pasangan pengamatan (n)
n = len(data)

# Menghitung derajat kebebasan (df)
df = n - 2

# Menghitung nilai t-tabel pada taraf signifikansi 0.1
t_tabel = t.ppf(1 - 0.1/2, df)

# Menghitung nilai r-tabel
r_tabel = t_tabel / math.sqrt(df + t_tabel**2)

print(f"Nilai r-tabel: {r_tabel:.4f}")

data['Skor Total'] = data.iloc[:, 2:].sum(axis=1)
data

# Menghitung korelasi Pearson antara setiap pertanyaan dengan kolom total
correlations = instruments.apply(lambda x: pearsonr(x, data['Skor Total'])[0])

# Menampilkan nilai r-hitung dan hasil validitas untuk masing-masing pertanyaan
print("Hasil Validitas Setiap Instrumen:")
for instrument, r_hitung in correlations.items():
    # Menampilkan nilai r-hitung
    print(f"{instrument} - r-hitung: {r_hitung:.4f}")

    # Uji validitas
    if abs(r_hitung) > r_tabel:
        print(f"{instrument} - Valid\n")
    else:
        print(f"{instrument} - Tidak Valid\n")

# Menentukan aspek-aspek
aspects = {
    'Flexibility and efficiency of use': [
        'Saya merasa informasi mengenai status dan proses pada aplikasi KAI Access sangat jelas',
        'Saya merasa saya mendapatkan umpan balik yang memadai mengenai tindakan yang dilakukan di dalam aplikasi'
    ],
    'Match between system and the real world': [
        'Saya merasa istilah dan bahasa yang digunakan dalam aplikasi KAI Access sesuai dengan yang diharapkan',
        'Saya merasa tindakan dan proses di aplikasi KAI Access konsisten dengan apa yang saya harapkan di dunia nyata tentang perjalanan menggunakan kereta api'
    ],
    'User control and freedom': [
        'Saya merasa memiliki kendali penuh ketika menggunakan aplikasi KAI Access',
        'Saya bisa dengan mudah mengembalikan tindakan jika terjadi kesalahan dalam menggunakan aplikasi KAI Access'
    ],
    'Consistency and standards': [
        'Elemen desain dan interaksi di seluruh aplikasi KAI Access sudah cukup konsisten',
        'Aplikasi KAI Access mengikuti standar dan konvensi desain yang sudah umum diterima'
    ],
    'Error prevention': [
        'Terdapat fitur dalam aplikasi KAI Access yang membantu mencegah kesalahan',
        'Saya sering melakukan tindakan yang tidak diinginkan ketika menggunakan aplikasi KAI Access'
    ],
    'Recognition rather than recall': [
        'Fungsi dan pilihan dalam aplikasi KAI Access mudah dikenali',
        'Saya perlu mengingat banyak informasi saat menggunakan aplikasi KAI Access'
    ],
    'Flexibility and efficiency of use': [
        'Saya perlu pintasan atau fitur lanjutan dalam aplikasi KAI Access yang memungkinkan saya menggunakan aplikasi dengan lebih efisien',
        'Aplikasi KAI Access dapat disesuaikan agar lebih sesuai dengan kebutuhan dan preferensi saya'
    ],
    'Aesthetic and minimalist design': [
        'Menurut saya desain visual aplikasi KAI Access sudah cukup menarik dan rapi',
        'Desain visual aplikasi KAI Access membantu saya fokus pada apa yang ingin saya lakukan'
    ],
    'Help users recognize, diagnose, and recover from errors': [
        'Pesan error di aplikasi KAI Access jelas dan mudah dimengerti',
        'Pesan error memberikan panduan yang bermanfaat tentang cara mengatasi masalah tersebut'
    ],
    'Help and documentation': [
        'Bantuan dan dokumentasi di dalam aplikasi KAI Access mudah ditemukan dan digunakan',
        'Dokumentasi memberikan instruksi yang jelas dan ringkas'
    ]
}
# Menghitung Cronbach's Alpha untuk setiap aspek
for aspect, instruments_list in aspects.items():
    alpha_result = reliability.cronbach_alpha(data[instruments_list])
    print(f"\nAspek: {aspect}")
    print("Hasil Reliabilitas Setiap Instrumen:")
    
    # Menampilkan nilai Cronbach's Alpha
    print(f"Cronbach's Alpha: {alpha_result['alpha']:.4f}")
    
    # Uji reliabilitas
    if alpha_result['alpha'] > 0.6:  # Sesuaikan dengan batasan yang diinginkan
        print(f"Aspek {aspect} - Reliabel\n")
    else:
        print(f"Aspek {aspect} - Tidak Reliabel\n")
