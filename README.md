# IF3230-K04-nubesx100-CUDA

Matrix Convolution with CUDA

## Skema Paralelisasi CUDA

### Perhitungan Konvolusi

Untuk perhitungan konvolusi pada tahap pertama akan dibuat terlebih dahulu grid 2 dimensi dan block 1 dimensi, dimana untuk grid x bernilai jumlah konvolusi yang ada dibagi dengan nilai statik BLOCK_SIZE yaitu 1024 dan untuk grid sumbu y bernilai jumlah input matrix yang ada. Untuk block nya sendiri karena 1 dimensi maka untuk block sumbu x nya bernilai statik BLOCK_SIZE yaitu 1024. Secara garis besar skema yang dibuat adalah 1 thread akan menghitung hasil 1 konvolusi.Untuk fungsi utamanya sendiri, akan di passing melalui parameter berupa semua input matrix, kernel, dan output matrix beserta semua ukurannya, lalu didalam fungsinya akan diambil index output yang nantinya akan ditaruh hasil konvolusinya. Akan dilakukan perulangan sebanyak 2 for loop untuk melakukan loop pada kernelnya dan menghitung hasil konvolusinya. Sebagai informasi tambahan semua matrix input, kernel, output akan dibuat menjadi representasi 1 dimensi.

### Pencarian Range Data

Perhitungan pencarian range data dilakukan dengan menggunakan data dalam bentuk larik 1 dimensi yang dihasilkan dari perhitungan konvolusi. Pada GPU dan CPU, akan dialokasikan memori untuk menampung keluaran dari pencarian range data sebanyak jumlah matriks input yang diberikan pengguna. Setelah itu, akan digunakan _grid_ dengan ukuran yang merupakan hasil ceiling dari ukuran matriks dan BLOCK*SIZE, dengan dimensi \_block* yang sama. Setiap _thread_ akan mengisi satu buah entri secara berurutan pada larik keluaran `(0, 1, ..., n_matrix - 1)`. Dalam hal ini, akan dicari _offset_ awal untuk setiap _thread_, dan dengan _offset_ ini maka _thread_ akan melakukan pencarian nilai maksimum dan minimum secara serial dari _offset_ tersebut hingga sebanyak jumlah elemen hasil konvolusi untuk sebuah matriks.

### Sorting

Sorting dilakukan dalam maksimal dua tahap. Pada tahap pertama, merge sort dilakukan dengan jumlah thread sebesar BLOCK_SIZE dan jumlah block sebesar ceiling num_matrix dibagi BLOCK_SIZE. Setiap block menangani BLOCK_SIZE elemen, dan satu thread menangani merge sort antara dua array. Setiap block mengalami maksimal log2(BLOCK_SIZE) pass, dimana pass pertama menggunakan maksimal BLOCK_SIZE thread untuk mengurut total BLOCK_SIZE array berukuran 1 elemen. Pass kedua menggunakan maksimal BLOCK_SIZE/2 thread untuk mengurut total BLOCK_SIZE/2 array berukuran 2 elemen, dan seterusnya. Pada tahap kedua, merge sort dilakukan untuk menggabungkan hasil pengurutan setiap block. Tahap kedua hanya dilakukan jika jumlah block pada tahap pertama lebih dari satu. Pada tahap kedua, terdapat hanya satu block dengan jumlah thread maksimal sebesar jumlah block pada tahap pertama. Pada pass pertama, maksimal N_BLOCK thread menggabungkan dua array berukuran BLOCK_SIZE, pada pass kedua, maksimal jumlah block dibagi 2 thread menggabungkan dua array berukuran BLOCK_SIZE\*2, dan seterusnya.

## Analisis Eksekusi Terbaik

Untuk setiap test case, eksekusi program paralel selesai dalam waktu yang lebih cepat daripada program serial. Speedup TC1 adalah 7.514x ekskusi serial, speedup TC2 adalah 54.698x, speedup TC3 adalah 22.376x, dan speedup TC4 adalah 22.250x. Speedup terbaik diperoleh pada test case 2, dengan parameter test adalah kernel ukuran 18x12 dan 101 matriks berukuran 100x100. Ini karena tingkat paralelisasi yang cukup baik, dari 1.64*10^8 operasi untuk konvolusi secara serial menjadi 216 operasi per thread. Pada TC2 juga, overhead pembuatan blok tidak setinggi pada TC3 atau TC4. Dalam TC2, jumlah blok pada grid sumbu y yang harus dibuat sebanyak 101, dan jumlah blok pada grid sumbu x yang harus dibuat sebanyak 8. Dibandingkan dengan 666 blok pada sumbu y TC3 dan 5000 blok pada sumbu y TC4, overhead untuk TC2 akan jauh lebih rendah. Maka dari itu, TC2 memiliki speedup terbaik antara test case lain.

## Perbandingan Hasil Eksekusi Program Secara Serial dan Paralel

Dari perbandingan antara hasil eksekusi program yang dilakukan secara serial serta program yang dieksekusi secara paralel, dapat dibuktikan bahwa kedua program memberikan hasil yang sama. Dengan demikian, dapat dikatakan bahwa program paralel yang dibuat telah mengatasi permasalahan-permasalahan yang mungkin muncul pada aplikasi paralel dan telah dibangun dengan benar. Selain itu, jika dilihat pada tabel di bawah, dapat dibuktikan bahwa program paralel memberikan peningkatan performansi program secara signifikan, yang _rate_-nya meningkat secara signifikan ketika ukuran permasalahan bertambah. Dengan demikian, dapat disimpulkan bahwa program paralel telah dibangun dengan baik dan benar.

## Eksperimen Variasi Eksekusi Program

```shell
| TC  | CUDA (second)| Serial (second) |
| --- | ------------ | -------------   |
| 1   | 0.001277     | 0.009592        |
| 2   | 0.013848     | 0.757452        |
| 3   | 0.032927     | 0.736760        |
| 4   | 0.438060     | 9.746812        |
| a   | 0.002916     | -               |
| b   | 0.000182     | -               |
| c   | 0.001897     | -               |

a = 10000 matriks, dengan ukuran kernel dan setiap matriks masing-masing 1x1 dan 1x1.
b = 1 matriks, dengan ukuran kernel dan setiap matriks masing-masing 1x1 dan 100x100.
c = 1 matriks, dengan ukuran kernel dan setiap matriks masing-masing 100x100 dan 100x100.
```

Untuk ketiga kasus tambahan yang ada, maka dapat diketahui bahwa kasus kedua memberikan waktu eksekusi yang paling cepat di antara ketiganya, disusul oleh kasus ketiga dengan selisih waktu yang tipis, dan diakhiri dengan kasus pertama dengan waktu yang lebih lama. Hal ini dikarenakan

## Author

1. 13519180 Karel Renaldi
2. 13519185 Richard Rivaldo
3. 13519205 Muhammad Rifat Abiwardani
