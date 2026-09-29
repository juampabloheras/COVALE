# Atlas licenses, citations, and modification notices

The registry records source URLs, citations, and license identifiers for every
bundled volume. This document groups the corresponding attribution and
modification notices.

## Digital 3D Brain MRI Arterial Territories Atlas

Bundled volumes:

- `arterialatlas.nii.gz`
- `arterialatlas-level2.nii.gz`

Source: [NITRC Arterial Atlas](https://www.nitrc.org/projects/arterialatlas)

License: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)

Citation:

> Liu CF, Hsu J, Xu X, Kim G, Sheppard S, Meier E, Miller MI, Hillis AE,
> Faria AV. Digital 3D Brain MRI Arterial Territories Atlas. Scientific Data.
> 2023;10:74. https://doi.org/10.1038/s41597-022-01923-0

The files are redistributed in the common COVALE MNI152 package layout.

## FreeSurfer

Bundled upstream-derived volumes:

- `aparc-a2009s-aseg.nii.gz`
- `aseg-auto.nii.gz`
- `lh-ribbon.nii.gz`
- `wm-asegedit.nii.gz`
- `wmparc.nii.gz`
- `wmparc-aligned-to-orig-h1.nii.gz`
- `wmparc-aligned-to-orig-h2.nii.gz`
- `wmparc-aligned-to-orig-h3.nii.gz`
- `wmparc-aligned-to-orig-h4.nii.gz`
- `nextbrain-left-right-merged.nii.gz`

Source: [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)

License: [FreeSurfer Software License Agreement, Version 1.0 (February 2011)](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)

General citation:

> Fischl B. FreeSurfer. NeuroImage. 2012;62(2):774-781.
> https://doi.org/10.1016/j.neuroimage.2012.01.021

The H1-H4 volumes are COVALE-authored hierarchy and alignment derivatives of
FreeSurfer `wmparc`. Their names, hierarchy definitions, alignment, compression,
and package layout differ from the upstream output.

The bundled NextBrain volume is a COVALE left-right merged derivative and was
resampled to the common 1 mm grid with nearest-neighbor interpolation. See
[`nextbrain_resampling.json`](nextbrain_resampling.json) for exact
transformation metadata and excluded labels.

### Destrieux parcellation

The `aparc-a2009s-aseg.nii.gz` volume also uses the Destrieux parcellation:

> Destrieux C, Fischl B, Dale A, Halgren E. Automatic parcellation of human
> cortical gyri and sulci using standard anatomical nomenclature. NeuroImage.
> 2010;53(1):1-15. https://doi.org/10.1016/j.neuroimage.2010.06.010

### Automated segmentation

The automated segmentation and white-matter products also derive from:

> Fischl B, et al. Whole brain segmentation: automated labeling of
> neuroanatomical structures in the human brain. Neuron.
> 2002;33(3):341-355.
> https://doi.org/10.1016/S0896-6273(02)00569-X

## NextBrain

Source: [FreeSurfer histological atlas segmentation](https://surfer.nmr.mgh.harvard.edu/fswiki/HistoAtlasSegmentation)

Citation:

> Casamitjana A, et al. A probabilistic histological atlas of the human brain
> for MRI segmentation. Nature. 2025.
> https://doi.org/10.1038/s41586-025-09708-2

The bundled derivative is distributed under the FreeSurfer Software License
Agreement identified above.

## COVALE-authored contextual masks

Bundled volumes:

- `midline.nii.gz`
- `cistern-segmentations.nii.gz`
- `tentorium.nii.gz`

Source: [COVALE](https://github.com/juampabloheras/COVALE)

License: [MIT](../../LICENSE)

Citation:

> COVALE contributors. COVALE: Compositional Open-Vocabulary Anatomical
> Localization Evaluation, v0.1.3.
> https://github.com/juampabloheras/COVALE
