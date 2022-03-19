# IF3230-K04-nubesx100-CUDA

Matrix Convolution with CUDA

## Skema Paralelisasi CUDA

## Analisis Eksekusi Terbaik

## Perbandingan Hasil Eksekusi Program Secara Serial dan Paralel

Dari perbandingan antara hasil eksekusi program yang dilakukan secara serial serta program yang dieksekusi secara paralel, dapat dibuktikan bahwa kedua program memberikan hasil yang sama. Dengan demikian, dapat dikatakan bahwa program paralel yang dibuat telah mengatasi permasalahan-permasalahan yang mungkin muncul pada aplikasi paralel dan telah dibangun dengan benar. Selain itu, jika dilihat pada tabel di bawah, dapat dibuktikan bahwa program paralel memberikan peningkatan performansi program secara signifikan, yang _rate_-nya meningkat secara signifikan ketika ukuran permasalahan bertambah. Dengan demikian, dapat disimpulkan bahwa program paralel telah dibangun dengan baik dan benar.

## Eksperimen Variasi Eksekusi Program

```shell
| TC  | CUDA (second)| Serial (second) |
| --- | ------------ | -------------   |
| 1   | 0.001277            | 0.009592 |
| 2   | 0.013848            | 0.757452 |
| 3   | 0.032927            | 0.736760 |
| 4   | 0.438060            | 9.746812 |
| a   | 0.002916            | -        |
| b   | 0.000182            | -        |
| c   | 0.001897            | -        |

a = 10000 matriks, dengan ukuran kernel dan setiap matriks masing-masing 1x1 dan 1x1.
b = 1 matriks, dengan ukuran kernel dan setiap matriks masing-masing 1x1 dan 100x100.
c = 1 matriks, dengan ukuran kernel dan setiap matriks masing-masing 100x100 dan 100x100.
```

Untuk ketiga kasus tambahan yang ada, maka dapat diketahui bahwa kasus kedua memberikan waktu eksekusi yang paling cepat di antara ketiganya, disusul oleh kasus ketiga dengan selisih waktu yang tipis, dan diakhiri dengan kasus pertama dengan waktu yang lebih lama. Hal ini dikarenakan

## Author

1. 13519180 Karel Renaldi
2. 13519185 Richard Rivaldo
3. 13519205 Muhammad Rifat Abiwardani
