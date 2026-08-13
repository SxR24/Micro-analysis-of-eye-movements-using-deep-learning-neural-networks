# Third-party code and models

This project incorporates work by others. Each component is listed below with
its origin, licence, and the citation its authors ask for.

---

## RITnet — segmentation model, weights, and architecture

**Files in this repository derived from RITnet:**

| file | relationship to the original |
|---|---|
| `src/ritnet/densenet.py` | unmodified `DenseNet2D` model definition |
| `models/best_model.pkl` | unmodified pretrained checkpoint (248,900 parameters) |
| `src/ritnet/ritnet_run.py` | written for this project, but loads and runs the above |

**Source:** https://github.com/AayushKrChaudhary/RITnet

**Licence:** MIT. The full notice is reproduced below as required by its terms.

```
The MIT License

Copyright (c) 2019 Aayush Chaudhary, Rakshit Kothari, Manoj Acharya,
Shusil Dangi, Nitinraj Nair, Reynold Bailey, Christopher Kanan,
Gabriel Diaz, and Jeff Pelz

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and
associated documentation files (the "Software"), to
deal in the Software without restriction, including
without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom
the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

**Citation requested by the authors:**

```bibtex
@inproceedings{chaudhary2019ritnet,
  title={RITnet: real-time semantic segmentation of the eye for gaze tracking},
  author={Chaudhary, Aayush K and Kothari, Rakshit and Acharya, Manoj and
          Dangi, Shusil and Nair, Nitinraj and Bailey, Reynold and
          Kanan, Christopher and Diaz, Gabriel and Pelz, Jeff B},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision
             Workshop (ICCVW)},
  pages={3698--3702},
  year={2019},
  organization={IEEE}
}
```

The DenseNet2D architecture in `densenet.py` is itself a simplified DenseNet
with U-Net structure, credited by its author (Shusil Dangi) to
https://github.com/ShusilDangi/DenseUNet-K

RITnet was trained on the OpenEDS dataset (Facebook Reality Labs). No training
was performed in this project; the published weights are used for inference
only.

---

## Irisometry — ocular torsion measurement

**`src/irisometry/ocular.py` is a derivative work.** It began as a port of the
MATLAB irisometry implementation in the Strauch/Naber lineage, obtained via
collaborators at the University of Applied Sciences Upper Austria and Utrecht
University, and follows the structure and logic of that original. It is a
translation into Python, not an independent reimplementation.

Substantial parts have since been changed for this project, and those parts are
original work: reference-anchored Lucas-Kanade tracking, feature gating by a
segmentation mask, the orthogonal Procrustes rotation estimator with Tukey
reweighting, sticky feature retirement, and the export of raw per-feature
trajectories and a per-frame fit residual. The overall design, the pupil
detection and blink logic, and the inner/outer annulus split derive from the
original.

**Licence status: unresolved.** The original was shared privately for research
use and carries no licence grant that the author of this repository is aware of.
No permission to redistribute a derivative work has been obtained in writing.
Until that permission is confirmed by the original authors, this file should be
treated as not licensed for redistribution, and anyone wishing to reuse it
should contact them rather than relying on the LICENSE file at the root of this
repository, which covers only original work.

**Relevant literature:**

- Strauch, C. and Naber, M. (2022) — irisometry / ocular torsion measurement
- Ivins, J.P. and Porrill, J. (1998) — iris pattern tracking for torsion
- Sadeghi, R. et al. (2024) — OpenIris



---

## Datasets

No third-party image data is distributed here. The eye videos under `data/raw/`
are recordings made for this project and are excluded from version control.
