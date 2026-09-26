# Smartwall WPT Simulation

Repositori ini memuat kode sumber (simulasi analitik, pemodelan numerik, dan Monte Carlo) untuk memvalidasi performa dari sistem **Smartwall Wireless Power Transfer (WPT)**. Proyek ini mendemonstrasikan integrasi *Coupled-Mode Theory* (CMT) pada resonansi medan-dekat dan *Phase-Conjugate Beamforming* pada metasurface untuk mengoptimalkan efisiensi transfer daya nirkabel di dalam ruangan.

## Struktur Repositori (Open Science)

Struktur kode ini didesain selaras 100% dengan turunan matematis pada makalah, menjamin reproduktifitas penuh bagi *peer-reviewer* dan peneliti lain:

```text
Smartwall-WPT-Simulation/
├── LICENSE                          # Teks lengkap Apache License 2.0 (Melindungi paten, mendukung Open Science)
├── README.md                        # Panduan instalasi dan deskripsi file
├── requirements.txt                 # Dependensi pustaka Python
│
├── data/
│   └── synthetic_wpt_baseline_comparison.csv   # Dataset statistik (9.000 trials) dari simulasi Monte Carlo
│
└── src/
    ├── cmt_resonant_splitting.py    # Perhitungan integral Neumann eliptik & bifurkasi CMT (Frequency Splitting)
    ├── fresnel_calc.py              # Radiasi zona Fresnel medan-dekat & batas Friis
    ├── ris_beamforming.py           # Perhitungan Phase-Conjugate Beamforming & model loss varaktor (Abeywickrama)
    ├── rectenna_model.py            # Pemodelan SPICE untuk dioda Schottky non-linear & sudut konduksi
    ├── monte_carlo_sim.py           # Eksekusi Monte Carlo N=9.000 dengan mempertimbangkan efek shadow fading
    └── ddpg_agent.py                # [FUTURE WORK] Eksperimen kontroler adaptif berbasis Deep Reinforcement Learning
```

## Persyaratan (Requirements)
Disarankan menggunakan Python 3.9+. Instal semua dependensi melalui:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan (Usage)
Untuk menghasilkan ulang (*reproduce*) dataset statistik dan grafik efisiensi, jalankan simulasi utama:
```bash
python src/monte_carlo_sim.py
```

## Catatan Ilmiah (Scientific Disclaimer)
Skrip utama penentu fase RIS (Reconfigurable Intelligent Surface) dipresentasikan di dalam `ris_beamforming.py` yang menggunakan kalkulasi analitis murni. Adanya modul *Reinforcement Learning* (seperti `ddpg_agent.py`, jika dilampirkan) hanyalah bagian dari *future work* untuk kontrol adaptif dan **bukan** metode pembuktian fundamental dalam makalah saat ini.

## Sitasi (Citation)
Jika Anda menggunakan kode ini dalam penelitian Anda, mohon sitasi makalah kami:
```bibtex
@article{wibisono2026smartwall,
  title={Physics-Grounded Smartwall Wireless Power Transfer: Multi-Frequency Metasurface Beamforming and Nonlinear Rectenna Modeling for Indoor Energy Autonomy},
  author={Wibisono, Ahmad Rifa'i and Pratama, Ridho Aria and Putra, Ardika Elyanza},
  journal={IEEE Access},
  year={2026}
}
```

## Hak Cipta & Lisensi
Hak Cipta © 2026 Ahmad Rifa'i Wibisono (UHAMKA).  
Repositori ini dirilis di bawah **Apache License 2.0**. Lihat file [LICENSE](LICENSE) untuk detail penuh.
