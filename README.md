# carla-data-acquisition

Generator dataset sintetis dari CARLA 0.9.16 untuk melatih YOLOPX: deteksi mobil,
drivable area, dan lane line. Label kotak 2D dibuat dengan algoritma **CarFree**
(Jang, Lee & Kim, *Applied Sciences* 2022, 12, 281), lalu disusun dalam format folder
YOLOPX / BDD100K sehingga bisa langsung dipakai untuk training.

Data bisa direkam dengan dua cara:

- **manual**: mobil dikendarai sendiri dengan keyboard (WASD), rekaman dinyalakan dan
  dimatikan dengan tombol R;
- **otomatis**: skenario menyalip di jalan tol Town04 dijalankan oleh skrip.

Keduanya menulis data mentah dengan format yang sama, jadi tahap pembuatan label
sesudahnya tidak peduli data itu berasal dari mana.

## Isi repo

| file | fungsi |
|---|---|
| `rig.py` | konfigurasi kendaraan ego dan kamera, plus fungsi yang dipakai bersama |
| `manual_drive.py` | kendarai Dodge Charger dengan WASD, tekan R untuk merekam |
| `spawn_traffic.py` | spawn kendaraan lain (Traffic Manager), terpisah dari skrip rekam |
| `capture_overtaking.py` | rekaman otomatis skenario menyalip di Town04 |
| `run_capture.sh` | menjalankan `capture_overtaking.py` dan me-restart server kalau crash |
| `carfree_gt.py` | label kotak 2D (Algoritma 1 sampai 3 CarFree) |
| `seg_gt.py` | mask drivable area dan lane line (dua versi: marka asli dan tersambung) |
| `build_yolopx_dataset.py` | menyusun folder dataset YOLOPX |
| `split_dataset.py` | membagi train / val / test |
| `verify_yolopx_format.py` | mengecek dataset persis seperti loader YOLOPX membacanya |
| `visualize_boxes.py` | menggambar kotak dan mask di atas gambar untuk dicek manual |
| `test_carfree_gt.py` | self-test algoritma, tidak butuh CARLA |

## Konfigurasi kendaraan dan sensor

Posisi sensor mengikuti tata letak kamera KITTI (cam2): satu kamera depan, di garis
tengah mobil, 1,65 m di atas jalan, menghadap lurus ke depan. Rig ini sama dengan yang
dipakai di [carla-ads-overtaking](https://github.com/SywlAchmd/carla-ads-overtaking).

| parameter | nilai |
|---|---|
| kendaraan ego | `vehicle.dodge.charger_2020` |
| panjang x lebar | 5,008 x 1,882 m |
| wheelbase L | 3,044 m |
| sumbu roda belakang ke origin aktor | 1,433 m |
| sudut roda maksimum | 70,0° (1,222 rad) |
| batas kemudi yang dipakai | 28,6° (0,5 rad) |
| massa | 1920 kg |

| sensor | spesifikasi | dipakai untuk |
|---|---|---|
| kamera RGB | 1280 x 720, FOV horizontal 90° | gambar dataset |
| kamera depth | 1280 x 720, FOV 90°, titik yang sama | validasi, opsional (`--no-depth`) |
| kamera instance segmentation | 1280 x 720, FOV 90°, titik yang sama | sumber label saja, tidak ikut dataset |
| collision sensor | event-based | dicatat di metadata, untuk evaluasi |

Kamera dipasang 1,68 m di depan sumbu roda belakang dan 1,65 m di atas jalan. CARLA
menaruh origin aktor di tengah bodi, di permukaan tanah, jadi di koordinat kendaraan
posisinya menjadi `x = 1,68 - 1,433 = 0,247 m`, `y = 0`, `z = 1,65 m`, tanpa rotasi.
Semua sensor kamera berada di transform yang sama sehingga piksel RGB, depth, dan
instance saling sejajar 1:1.

Intrinsik (ditulis ke `calib/seqXXXX.txt`): `fx = fy = 640`, `cx = 640`, `cy = 360`.
FOV vertikal mengikuti rasio 16:9, yaitu sekitar 58,7°.

Setiap kali dijalankan, `manual_drive.py` mengukur ulang posisi kamera dari posisi roda
fisik dan mencetaknya, misalnya:

```
vehicle.dodge.charger_2020 on Town04 | camera 1.641 m above road, 1.680 m ahead of rear axle | wheelbase 3.044 m | steer capped at 0.409 (28.6 of 70.0 deg)
```

Kalau angka ini meleset (misalnya setelah ganti versi CARLA), ubah konstanta di `rig.py`.

Kap mesin Charger ikut terlihat di sekitar 17% bagian bawah gambar. Itu memang akibat
posisi kamera tersebut, dan mobil ego tidak pernah diberi label.

## Instalasi

Butuh CARLA 0.9.16 dan Python 3.8 sampai 3.10.

```bash
pip install -r requirements.txt
```

`capture_overtaking.py` memakai `agents.navigation.controller` bawaan CARLA. Set
`CARLA_ROOT` ke folder instalasi CARLA supaya modul itu ketemu:

```bash
export CARLA_ROOT=~/CARLA_0.9.16
```

Cek algoritma tanpa server:

```bash
python3 test_carfree_gt.py
```

## Merekam secara manual

Butuh tiga terminal.

**Terminal 1, server CARLA:**

```bash
cd $CARLA_ROOT
./CarlaUE4.sh -quality-level=Epic       # tambah -RenderOffScreen kalau tanpa monitor
```

**Terminal 2, mobil ego:**

```bash
python3 manual_drive.py --map Town04
```

Akan muncul jendela preview dari kamera depan.

| tombol | aksi |
|---|---|
| W | gas |
| S | rem |
| A / D | belok kiri / kanan |
| Q | ganti maju / mundur |
| SPACE | rem tangan |
| R | mulai / berhenti merekam |
| ESC | keluar |

Setiap kali R ditekan untuk mulai, dibuat folder baru `out/raw/run_XXXX/`. Selama
merekam, di pojok kiri atas muncul tulisan merah `REC run_XXXX` beserta jumlah frame.
Simulasi berjalan 20 Hz dan disimpan 1 dari 2 tick (10 Hz). Secara default frame yang
tidak memuat mobil lain tidak disimpan; pakai `--keep-empty` kalau butuh sampel negatif.

Kemudi dibatasi 0,5 rad supaya data manual tetap berada di rentang kemudi yang sama
dengan kontroler MPC. Gas dan rem naik bertahap, dan setir bergerak dengan kecepatan
tetap, jadi menekan A atau D sebentar tidak langsung membanting setir.

**Terminal 3, kendaraan lain:**

```bash
python3 spawn_traffic.py -n 60
```

Jalankan setelah `manual_drive.py` sudah hidup. Skrip ini hanya memunculkan kendaraan
dan menyerahkannya ke Traffic Manager, tidak ikut menjalankan simulasi. Yang di-spawn
hanya mobil penumpang roda empat, supaya semua objek tetap satu kelas `car`.

**Urutan berhenti:** matikan `spawn_traffic.py` dulu (Ctrl+C), baru `manual_drive.py`
(ESC). Traffic Manager hidup di dalam proses `manual_drive.py`, dan server CARLA bisa
segfault kalau Traffic Manager mati saat masih memegang kendaraan. Kedua skrip sudah
menangani kasus itu, tapi urutan di atas tetap yang paling aman.

Opsi lain `manual_drive.py`:

| opsi | default | keterangan |
|---|---|---|
| `--map` | map yang sedang aktif | misalnya `Town04`, `Town10HD_Opt` |
| `--weather` | `ClearNoon` | preset `carla.WeatherParameters` |
| `--spawn` | acak | indeks spawn point |
| `--out` | `out` | folder output |
| `--fps` / `--record-every` | 20 / 2 | frekuensi simulasi / simpan 1 dari N tick |
| `--no-depth` | | tidak merekam depth (hemat sekitar 400 kB per frame) |
| `--keep-empty` | | simpan juga frame tanpa mobil |
| `--no-parked` | | hilangkan mobil parkir bawaan map (hanya map `_Opt`) |

Soal `--no-parked`: di beberapa kota ada mobil parkir yang merupakan bagian dari map,
bukan aktor. Pikselnya ber-tag `Car`, tapi tidak punya kotak 3D sehingga tidak pernah
mendapat label, dan model akan belajar bahwa mobil itu latar belakang. Town04 praktis
tidak punya mobil parkir; untuk kota lain pakai versi `_Opt` beserta opsi ini.

## Merekam secara otomatis

```bash
python3 capture_overtaking.py --total-frames 4000 --out out --follow
```

atau, supaya tetap lanjut kalau server crash:

```bash
./run_capture.sh 5000 out
```

Skrip ini mencari ruas lurus berlajur banyak di Town04 (setiap kandidat ditelusuri
200 m ke depan, ditolak kalau arah berubah lebih dari 2° atau lajur sebelahnya hilang),
lalu memutar tiga jenis skenario:

| skenario | yang terlihat kamera |
|---|---|
| `overtake_left` (bobot x2) | mendekat, pindah ke kiri, sejajar (mobil terpotong di tepi kanan gambar), kembali ke lajur |
| `overtake_right` | sama, dicerminkan |
| `being_overtaken` | NPC menyalip ego lalu masuk ke depan, mobil makin jauh |

`being_overtaken` diperlukan karena kamera depan tidak bisa melihat mobil yang baru saja
disalip. Hanya dengan skenario ini fase sesudah menyalip ikut terekam.

Kedua mobil dikendalikan `VehiclePIDController` CARLA dengan target ID lajur, jadi
pindah lajur cukup dengan mengganti ID target. Waktu menyalip bisa diatur dengan tepat
tanpa Traffic Manager. Kendaraan latar tetap memakai Traffic Manager.

Parameter yang diacak per run: kecepatan mobil depan 30 sampai 42 km/jam, kecepatan
menyalip +16 sampai 30 km/jam, jarak awal 20 sampai 40 m, jarak mulai menyalip 12 sampai
20 m, jarak kembali ke lajur 14 sampai 22 m, jenis dan warna mobil NPC, sisi, dan ruas
jalan. Run berhenti kalau ego menabrak, jadi frame sesudah tabrakan tidak pernah masuk
dataset.

| opsi | default | keterangan |
|---|---|---|
| `--total-frames` | 4000 | terus membuat run baru sampai jumlah ini tercapai |
| `--traffic` | 12 | jumlah kendaraan latar per run |
| `--weathers` | `ClearNoon` | daftar dipisah koma, bergiliran per run |
| `--keep-empty` | | simpan frame tanpa mobil |
| `--depth` | | rekam depth juga |
| `--no-instance` | | hanya kamera semantic, persis seperti paper |
| `--seed` | 2024 | parameter run bisa direproduksi |
| `--resume` | | lanjutkan penomoran run yang sudah ada |

## Dari data mentah ke dataset

Langkahnya sama untuk rekaman manual maupun otomatis, dan keduanya boleh dicampur di
folder `out/raw` yang sama.

```bash
# 1. label kotak
python3 carfree_gt.py --raw out/raw --mask-source instance

# 2. mask drivable area + lane line
python3 seg_gt.py --raw out/raw

# 3. susun folder YOLOPX (--lanes marking atau continuous, lihat bagian lane line)
python3 build_yolopx_dataset.py --raw out/raw --dataset dataset_root --lanes marking

# 4. bagi train / val / test
python3 split_dataset.py --dataset dataset_root --val 0.1 --test 0.1

# 5. cek dengan aturan loader YOLOPX
python3 verify_yolopx_format.py --dataset dataset_root

# 6. lihat hasilnya
python3 visualize_boxes.py --dataset dataset_root --n 40 --masks --out check
```

Label dibuat di proses terpisah dari perekaman, sama seperti arsitektur CarFree
(`extract.py` saat simulasi, pembuatan label offline). Keuntungannya, label bisa dibuat
ulang dengan pengaturan lain tanpa perlu menjalankan simulasi lagi.

### Mengubah label tanpa merekam ulang

Data mentah (`rgb/`, `inst/`, `meta.jsonl`) tidak pernah diubah oleh tahap label. Kalau
aturan label diganti, cukup jalankan ulang langkah 1 sampai 5. `build_yolopx_dataset.py`
mengosongkan `all/` setiap kali dijalankan dan `split_dataset.py` mengosongkan folder
split-nya, jadi tidak ada label lama yang tertinggal.

Yang bisa diubah dari data yang sudah ada: sumber mask, cara cek visibilitas, batas
ukuran kotak, jarak maksimum (sampai 120 m), aturan `occluded`/`truncated`, logika
drivable area dan lane line, nama kelas, format JSON, dan rasio split.

Yang butuh rekam ulang: posisi, FOV, atau resolusi kamera; frame kosong yang tidak
disimpan (kecuali direkam dengan `--keep-empty`); mobil yang lebih jauh dari
`--max-distance` saat merekam; objek selain kendaraan (pejalan kaki tidak dicatat di
`meta.jsonl`); dan depth kalau direkam dengan `--no-depth`.

Gunakan `--mask-source instance`. Kap mobil ego selalu terlihat di bawah gambar dan di
peta semantic ikut ber-tag `Car`. Dengan mask semantic, mobil yang berada tepat di
depan bisa menyatu dengan kap itu.

## Struktur data mentah

```
out/raw/run_0000/
  run.json          peta, skenario, cuaca, parameter, posisi kamera, alasan run berhenti
  meta.jsonl        satu baris per frame: matriks kamera, kecepatan dan kontrol ego,
                    tabrakan, geometri lajur, 8 titik sudut kotak 3D setiap mobil
  rgb/000000.jpg
  inst/000000.png   instance segmentation, byte mentah dari CARLA
  depth/000000.png  depth CARLA (opsional)
  det/000000.json   hasil carfree_gt.py
  da/000000.png     hasil seg_gt.py
  ll/000000.png     hasil seg_gt.py, lane line versi marka
  ll_cont/000000.png hasil seg_gt.py, lane line versi tersambung
```

## Cara label kotak dibuat

Empat langkah CarFree:

1. **Transformasi koordinat.** Delapan sudut `actor.bounding_box` diubah dari koordinat
   kendaraan ke dunia, lalu ke kamera (sumbu UE diubah ke x kanan, y bawah, z depan),
   lalu diproyeksikan ke gambar dengan model pinhole. Sebelum diproyeksikan, 12 rusuk
   kotak dipotong terhadap bidang dekat 0,2 m. Tanpa langkah ini, mobil yang sedang
   sejajar dengan kamera (sebagian di belakang bidang kamera) menghasilkan kotak yang
   kacau, bukan kotak terpotong.
2. **Algoritma 1.** Nilai min/max dari titik hasil proyeksi menjadi kotak awal.
3. **Algoritma 2.** Piksel tengah kotak awal harus ber-kelas target di mask segmentasi.
   Kalau tidak, objek dianggap tertutup dan dibuang. `--visibility multi5` memakai lima
   titik (tengah dan empat sudut), perluasan yang disarankan di paper untuk objek yang
   tertutup sebagian.
4. **Algoritma 3.** Setiap sisi kotak digeser ke dalam sampai menyentuh piksel target.
   Sisi yang mentok di tepi gambar berhenti di situ, sehingga kotak yang terpotong tetap
   tepat sampai piksel.

Ada dua perbedaan dari pseudocode paper, keduanya disengaja dan dites di
`test_carfree_gt.py`:

- Penyusutan hanya memeriksa bagian kolom/baris yang ada **di dalam kotak**, bukan satu
  kolom penuh. Kalau satu kolom penuh, mobil lain yang berada di kolom yang sama akan
  menghentikan penyusutan terlalu cepat.
- Loop-nya divektorkan (`any()` + `flatnonzero`), bukan maju satu piksel per langkah.
  Hasilnya identik tetapi jauh lebih cepat.

Kamera instance CARLA menyimpan ID aktor sebagai **byte0 = byte tinggi, byte1 = byte
rendah, byte2 = tag semantic** di buffer mentahnya, kebalikan dari urutan yang tersirat
di dokumentasi. Kalau dibaca terbalik, semua mask per aktor kosong tanpa ada error.
`test_instance_id_decoding` mengunci urutan ini, dan `carfree_gt.py` melaporkan berapa
objek yang tidak punya piksel instance (angka yang besar berarti decoding-nya rusak).

Kanal R kamera instance identik dengan kamera semantic (dibandingkan pada 25 frame,
berbeda 1 piksel dari sekitar 23 juta), jadi kamera semantic terpisah tidak di-spawn
lagi. `load_tags()` membaca `sem/` kalau ada, dan kalau tidak ada mengambil dari kanal R
instance.

## Drivable area dan lane line

**Lane line** tersedia dalam dua versi, dan `seg_gt.py` selalu membuat keduanya:

| versi | folder mentah | isi |
|---|---|---|
| `marking` | `ll/` | marka jalan persis seperti yang dirender (tag semantic 24). Garis putus-putus tetap putus, dan marka di seberang median ikut terlabel. |
| `continuous` | `ll_cont/` | tepi setiap lajur di jalur ego digambar sebagai garis utuh dari geometri OpenDRIVE, sehingga setiap lajur terbentuk sebagai pita tertutup. Mirip gaya label lane BDD100K. |

Versi `continuous` diiris dengan piksel jalan sama seperti drivable area, jadi garis di
belakang mobil tetap tertutup. Tebal garis diatur dengan `seg_gt.py --ll-width`
(default 8 px). Versi ini hanya mencakup jalur searah ego; marka jalur seberang median
tidak ikut.

Pilih versi saat menyusun dataset. Untuk membuat dua dataset sekaligus dari data yang
sama, jalankan build dua kali ke folder yang berbeda:

```bash
python3 build_yolopx_dataset.py --raw out/raw --dataset dataset_marking --lanes marking
python3 build_yolopx_dataset.py --raw out/raw --dataset dataset_continuous --lanes continuous
```

Setelah itu jalankan `split_dataset.py` dengan `--seed` yang sama untuk keduanya, supaya
pembagian train/val/test-nya identik dan hasil kedua versi bisa dibandingkan langsung.

**Drivable area** tidak bisa diambil dari mask semantic. Di jalan tol dengan pembatas
tengah, kelas `Road` juga mencakup jalur arah berlawanan: permukaannya jalan, tetapi
tidak boleh dilalui. Karena itu setiap frame menyimpan geometri OpenDRIVE dari jalur ego
sendiri (lajurnya ditambah semua lajur searah di sebelahnya) sebagai titik tepi kiri dan
kanan setiap 6 m sampai 200 m ke depan. `seg_gt.py` mengubahnya menjadi poligon,
memproyeksikannya, lalu mengirisnya dengan piksel jalan dari segmentasi. Irisan itu
sekaligus memotong area ke jalan yang benar-benar terlihat, membuang bahu jalan, dan
melubangi area yang tertutup kendaraan, guard rail, atau pembatas.

`seg_gt.py` mencetak **horizon gap**, yaitu jumlah baris piksel jalan yang terlihat di
atas area berlabel. Kalau lebih dari 25 px, jangkauan geometri kurang jauh.

Kedua mask 8-bit dengan 0 untuk latar dan 255 untuk objek, sesuai loader YOLOPX yang
melakukan threshold di atas 1.

## Format label

Satu JSON per gambar di `det_annotations/`, mengikuti skema yang benar-benar dibaca oleh
`lib/dataset/bdd.py` di YOLOPX (`frames[0].objects`). Format BDD100K yang lebih baru
(`{"name", "labels"}`) akan menimbulkan `KeyError` di loader tersebut.

```json
{
  "name": "town04_seq0001_frame00012.jpg",
  "attributes": {"weather": "ClearNoon", "scene": "highway",
                 "timeofday": "daytime", "scenario": "manual"},
  "frames": [{"objects": [{
    "id": 342,
    "category": "car",
    "box2d": {"x1": 512.0, "y1": 210.0, "x2": 640.0, "y2": 300.0},
    "attributes": {"occluded": false, "truncated": true},
    "carla": {"distance": 18.4, "type_id": "vehicle.audi.tt",
              "visible_ratio": 0.71, "mask_pixels": 8190}
  }]}]
}
```

- `x2`/`y2` eksklusif, jadi `lebar = x2 - x1`. Koordinat selalu dibatasi ke
  `[0, 1280] x [0, 720]`.
- `truncated` bernilai true kalau kotak sebelum dipotong keluar dari gambar, atau
  sebagian kotak 3D berada di belakang kamera.
- `occluded` adalah heuristik: ada piksel kendaraan lain di dalam kotak, atau kotaknya
  terlalu kosong.
- Blok `carla` hanya untuk audit dan diabaikan YOLOPX.

## Konfigurasi YOLOPX

```python
_C.DATASET.DATAROOT  = '<dataset_root>/images'
_C.DATASET.LABELROOT = '<dataset_root>/det_annotations'
_C.DATASET.MASKROOT  = '<dataset_root>/da_seg_annotations'
_C.DATASET.LANEROOT  = '<dataset_root>/ll_seg_annotations'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET  = 'val'          # atau 'test'
_C.DATASET.ORG_IMG_SIZE = [720, 1280]
_C.num_seg_class = 2
```

Resolusi 1280 x 720 sama dengan `ORG_IMG_SIZE` bawaan YOLOPX, jadi tidak perlu resize.
`category: "car"` cocok dengan `id_dict_single` di `lib/dataset/convert.py`.

Folder `all/` hanya kumpulan semua frame yang dipakai oleh `split_dataset.py`. Isinya
hardlink sehingga tidak memakan ruang disk, tetapi kalau diarsipkan ukurannya jadi
dua kali lipat. Kecualikan `*/all/` saat membagikan dataset.

## Hal yang perlu diketahui

- **Split per frame.** Secara default `split_dataset.py` membagi per frame secara acak,
  sehingga frame yang berurutan dari satu run bisa masuk train dan val sekaligus. Itu
  cukup untuk mengecek pipeline, tetapi untuk angka mAP yang dilaporkan pakai
  `--by-sequence`.
- **Kotak kecil dibuang.** Kotak yang tingginya kurang dari 25 px dibuang
  (`--min-box-h`, default 25). Angka ini diambil dari benchmark deteksi KITTI, yang
  membagi objek menjadi tiga tingkat: Easy (tinggi kotak minimal 40 px), Moderate dan
  Hard (minimal 25 px). Objek di bawah 25 px tidak ikut dihitung di evaluasi KITTI.
  Dataset `yolopx_dataset_v2` juga dibuat dengan batas ini.

  Batasnya dalam piksel, bukan meter, jadi jarak yang setara bergantung pada panjang
  fokus. Di sini fy = 640 px, sedangkan KITTI sekitar 721 px. Diukur dari label v2
  (field `carla.distance`), kotak di dekat batas 25 px berjarak median 39,6 m, dengan
  rentang 32 sampai 55 m tergantung tinggi mobilnya. 95% dari semua kotak berada dalam
  39,6 m, dan yang terjauh 55,5 m.

  KITTI menandai objek yang terlalu kecil sebagai *DontCare*, sedangkan loader YOLOPX
  tidak punya konsep itu, sehingga objek tersebut dihapus dan model menganggapnya latar
  belakang. Kalau dataset perlu lebih besar, `--min-box-h 16` menjangkau sekitar 60 m.
  Di bawah itu, setelah YOLOPX mengecilkan gambar ke 640, mobil tinggal kurang dari 8 px
  (satu stride output) dan praktis tidak bisa dipelajari.
- **Persimpangan.** Geometri drivable area mengikuti cabang pertama dari `next()`. Di
  persimpangan, cabang lain tidak ikut berlabel. Untuk rekaman manual di kota, perhatikan
  hal ini saat mengecek hasil.
- **Tabrakan di mode manual** hanya dicatat (`collision` di `meta.jsonl`, `collisions` di
  `run.json`) dan rekaman tetap berjalan. Filter sendiri kalau frame sesudah tabrakan
  tidak diinginkan.
- **Ruang disk.** Sekitar 230 kB per frame tanpa depth dan sekitar 630 kB dengan depth.

## Referensi

- Jang, J., Lee, H., & Kim, J.-C. (2022). CarFree: Hassle-Free Object Detection Dataset
  Generation Using Carla Autonomous Driving Simulator. *Applied Sciences*, 12(1), 281.
- Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The
  KITTI vision benchmark suite. *CVPR 2012*. (tingkat Easy/Moderate/Hard dan batas
  tinggi kotak 40/25 px)
- Geiger, A., Lenz, P., Stiller, C., & Urtasun, R. (2013). Vision meets robotics: The
  KITTI dataset. *The International Journal of Robotics Research*, 32(11).
- CARLA Simulator 0.9.16, https://carla.org
- YOLOPX, https://github.com/jiaoZ7688/YOLOPX
