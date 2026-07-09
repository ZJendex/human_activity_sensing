# Radar Human Activity Sensing Datasets With Verified Links

Last updated: 2026-07-09.

This revision keeps only papers or dataset records where I found a direct dataset repository, DOI, download page, or official dataset access page. Papers with only a paper PDF, an abstract, or a "will be released" statement are omitted from the main lists.

NR means not reported in the linked paper, repository, or dataset page checked in this pass. Durations marked "derived" were computed from the paper's reported frames/segments and frame rate.

## 1. Actual Shared Datasets With Dataset-Link Citation

This list includes verified shared datasets even when the paper/repository did not provide enough information to compute total hours.

| Dataset / paper | Actual shared dataset link | Paper / project citation | Duration and frame-rate status | Radar / RF setup | Classes / activity type | Environments / notes |
|---|---|---|---|---|---|---|
| MM-Fi | [GitHub dataset repo](https://github.com/ybhbingo/MMFi_dataset) with cloud download links | [Project page](https://ntu-aiot-lab.github.io/mm-fi), [arXiv paper](https://arxiv.org/abs/2305.10345) | approx. 8.9 h derived; 10 Hz synchronized rate, 30 Hz native mmWave stream | TI IWR6843, 60-64 GHz, synchronized multimodal setup | 27 activities; pose/action sensing | 4 domains, 40 subjects |
| OPERAnet | [Figshare dataset](https://figshare.com/s/c774748e127dcdecc667) | [arXiv paper](https://arxiv.org/abs/2110.04239) | approx. 8 h reported; WiFi CSI/PWR at 1600 Hz, UWB at about 400 Hz and 195 Hz | WiFi CSI, Passive WiFi Radar, UWB, Kinect | 6 main daily activities plus background/localization/crowd records | 2 rooms, 6 participants |
| MMVR | [Zenodo dataset](https://doi.org/10.5281/zenodo.12611978) | [arXiv paper](https://arxiv.org/abs/2406.10708) | 6.58 h reported/derived; approx. 14.6 frames/s derived from 345k frames and 395 min | Multi-view high-resolution mmWave radar heatmaps | Pose, instance segmentation, and detection annotations | 6 rooms, 25 subjects |
| mmDoppler | [GitHub dataset repo](https://github.com/arghasen10/mmdoppler) | [arXiv paper](https://arxiv.org/abs/2407.21125) | 6.42 h reported; macro 5 FPS, micro 2 FPS | TI IWR1642BOOST, 77-81 GHz | 19 macro/micro activities | 3 rooms, 7 subjects |
| mRI | [GitHub dataset repo](https://github.com/sizhean/mri) | [Project page](https://sizhean.github.io/mri), [arXiv paper](https://arxiv.org/abs/2210.08394) | approx. 4.44 h derived; 10 Hz synchronized frame rate | TI IWR1443BOOST with RGB-D and IMU | 12 human actions with 3D pose labels | 1 laboratory setting, 20 subjects |
| DISC | [Dataset DOI](https://dx.doi.org/10.21227/2gm7-9z72) | [arXiv paper](https://arxiv.org/abs/2306.09469) | more than 2 h reported; 0.27 ms inter-packet time | 60 GHz IEEE 802.11ay SDR testbed | 5 activities; activity/gait/sparse sensing | 2 environments, 7 subjects |
| RadHAR | [GitHub dataset repo](https://github.com/nesl/RadHAR) | [ACM paper DOI](https://doi.org/10.1145/3349624.3356768) | 1.55 h reported from repo split durations; 30 Hz radar frames | TI IWR1443BOOST point-cloud radar | 5 activities: boxing, jack, jump, squats, walk | 1 laboratory setting |
| CI4R-MULTI3 | [GitHub dataset repo](https://github.com/ci4r/CI4R-MULTI3) with dataset download link | [SPIE paper DOI](https://doi.org/10.1117/12.2559155) | Total hours NR; frame rate NR in checked repo | Synchronized XeThru 10 GHz UWB impulse radar, Ancortek 24 GHz FMCW radar, and TI IWR1443BOOST 77 GHz FMCW radar | 11 activities/gaits; about 60 samples per class per sensor | 1 lab line-of-sight setup, 6 participants |
| University of Glasgow radar signatures of human activities | [University of Glasgow dataset DOI](https://doi.org/10.5525/gla.researchdata.848) | [Dataset record](https://researchdata.gla.ac.uk/848/) and related publications listed there | Total hours NR; recording FPS NR; chirp duration 1 ms | Ancortek 5.8 GHz FMCW radar, 400 MHz bandwidth, 1 ms chirp duration, Yagi antennas | 5 main activities plus optional simulated fall | Multiple named rooms/facilities across University of Glasgow, NG Homes, and Age UK West Cumbria |
| MARS: mmWave-based Assistive Rehabilitation System for Smart Healthcare | [GitHub dataset repo](https://github.com/sizhean/MARS) | [ACM paper DOI](https://doi.org/10.1145/3477003) | Total hours NR; comparison tables report 40k frames, but FPS was not verified in the linked repo | mmWave radar with synchronized Kinect data; exact radar board NR in checked repo metadata | 10 rehabilitation movements | 1 controlled rehabilitation/lab setting, 4 subjects |
| CubeLearn: End-to-End Learning for Human Motion Recognition from Raw mmWave Radar Signals | [GitHub dataset/code repo](https://github.com/zhaoymn/cubelearn) | [arXiv paper](https://arxiv.org/abs/2111.03976) | Total hours NR; comparison tables report about 1k frames, but FPS was not verified | TI IWR6843 mmWave radar, 3 TX / 4 RX | 6 motion classes | 1 controlled lab setting, 8 participants |
| DIAT-uRadHAR radar micro-Doppler dataset | [IEEE DataPort dataset link](https://ieee-dataport.org/documents/diat-%CE%BCradhar-radar-micro-doppler-signature-dataset-human-suspicious-activity-recognition) | [AIRHAR/RadMamba repo pointing to the dataset](https://github.com/lab-emi/AIRHAR), [RadMamba paper](https://arxiv.org/abs/2504.12039) | Total hours NR; frame rate NR in accessible sources | Non-continuous CW radar micro-Doppler dataset; detailed hardware metadata not accessible in this pass | Suspicious-activity recognition; class count NR in accessible sources | Environment count NR; access and metadata depend on IEEE DataPort page |

## Notes

- The ranked duration list is intentionally shorter than the broader dataset-link list because several public repositories do not report total recording time or enough frame-rate information to compute it reliably.
- For derived durations, I used only explicit frame/segment counts and explicit frame rates from the paper or repository.
- I did not include claimed-but-unverified releases such as MiliPoint, HuPR, DGHMesh, or CUHK-X full release in the main lists because I could not confirm currently downloadable dataset files or a usable dataset DOI/repository for them during this pass.
