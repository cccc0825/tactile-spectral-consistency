# Data and third-party attribution

## Cluster Haptic Texture Dataset

Michikuni Eguchi, Tomohiro Hayase, Yuichi Hiroi and Takefumi Hiraki. Dataset v5: https://doi.org/10.6084/m9.figshare.29438288.v5 . Associated article: https://doi.org/10.1038/s41597-026-06760-z . The recorded source metadata identifies CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The source release includes the public ZIP member index, not raw recordings. Processing averages duplicate timestamps, resamples central windows, estimates and aggregates axis-wise Welch spectra, and evaluates acquisition-condition holdouts. Downloaded ranges and archive-member CRCs do not establish a locally verified hash of the complete parent archive.

## Multimodal Tactile Texture Dataset

Bruno Monteiro Rocha Lima, Thiago Eustaquio Alves de Oliveira and Vinicius Prado da Fonseca. Mendeley v1: https://doi.org/10.17632/n666tk4mw9.1 . CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The associated article, https://doi.org/10.1016/j.dib.2023.109590 , also credits Venkata Naga Sai Siddhartha Danyamraju; dataset and article creator lists are distinct.

The code uses IMU channels, explicit filtering/windowing and the recorded eligibility criteria. Archive folders use speeds 30/35/40 mm/s; some descriptive source text differs. The two included prediction CSVs and derived specimen summaries are study outputs based on these attributed observations. No raw participant or sensor recordings are bundled. Public metadata is minimized to dataset identity, attribution, license and stable file URLs; generated compatibility HTML is not the original landing page.

## Reference code

The four files under `work/revision/vendor/cluster-official/` are pinned to commit `e05d6b022d127e24f73583146f0aa229c6934449`. Their original Apache-2.0 license and source notices are retained. The local ResNet34 experiment changes splits, preprocessing and training budget relative to that reference and does not reproduce its random-split benchmark.

## Pretrained initialization

The recorded ResNet34 dependency is https://download.pytorch.org/models/resnet34-b627a593.pth , SHA-256 `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`. The weight file and all trained checkpoints are excluded. The reference repository's source license is not a redistribution license for ImageNet images or a separate pretrained artifact.

## Study code

The remaining study-specific programs implement the methods and analysis described in the manuscript. Dependency packages retain their respective licenses. Public visibility alone does not establish an MIT, Apache or other project-wide license; none has been assigned to the study-specific code in this release.
