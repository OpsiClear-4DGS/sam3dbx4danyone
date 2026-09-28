# Third-party notices

The root [GNU AGPL-3.0-only license](../LICENSE) covers this adaptation and its original additions and modifications. Inherited 4DAnyone code retains Apache-2.0. Retained third-party code, model assets and dependencies have separate terms; the root license does not replace their licenses. The default `full` mode uses the Turbo adapter under **CC BY-NC-SA 4.0**, alongside Meta model/schema licenses and NVIDIA package/model terms.

License texts are retained under [`third_party/licenses`](../third_party/licenses), with retrieval sources and SHA-256 hashes in [SOURCES.json](../third_party/licenses/SOURCES.json). The [dependency inventory](dependency_licenses.json) records the installed third-party distributions in the validated Linux/Python 3.11 environment, including optional TensorRT and GUI packages, their upstream-declared license metadata and installed license-file paths. It describes the installed environment, not a grant covering all components or every possible platform resolver result.

Source and wheel packages include this notice, the inventory and retained license texts. The bundled SageAttention wheel retains its own license. Downloaded model weights are not embedded in the Python package.

## Upstream 4DAnyone

- Source: <https://github.com/ant-research/4DAnyone>
- Source revision used by this adaptation: `4f66efeea83c641ef7c57da7df9c33c4f3c726e4`
- Original code license: Apache-2.0, retained at [4DANYONE_LICENSE](../third_party/licenses/4DANYONE_LICENSE)
- Original authors and paper: [README attribution](../README.md#license-and-attribution)

The project history starts with a single release snapshot. This does not change upstream authorship or licensing. Original upstream code, copyright and attribution remain under their existing terms. This project's modifications and new workflow code use AGPL-3.0-only. Modified inherited Python files carry a notice identifying the adaptation; vendored components retain their own provenance records.

## SAM 3D Body and MHR

- Source: <https://github.com/facebookresearch/sam-3d-body>
- Revision: `b5c765a0d89d789985e186d396315e7590887b94`
- Installed source: `models/sam3d-body/source`
- SAM source/model license: [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/LICENSE)
- Local SAM license: [SAM3D_BODY_LICENSE](../third_party/licenses/SAM3D_BODY_LICENSE)
- MHR source: <https://github.com/facebookresearch/MHR>, Apache-2.0; [local license](../third_party/licenses/MHR_LICENSE)
- DINOv3 backbone: <https://github.com/facebookresearch/dinov3>, custom [DINOv3 license](../third_party/licenses/DINOV3_LICENSE)

The MHR and DINOv3 license snapshots were retrieved from the revisions recorded in `SOURCES.json`; these are license-source revisions, not claims that the GEM-X-exported tensors were built from those exact source commits.

The direct body decoder, configuration, MHR model, and DINOv3 backbone export
are downloaded from `nvidia/GEM-X` at revision
`5ccf5ca3746c3620aa4016114f069a5f6ae399cd`. That distribution uses the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/);
embedded upstream materials retain their own terms. The pinned distribution’s [code license (Apache-2.0)](../third_party/licenses/GEMX_LICENSE) and [model card](../third_party/licenses/GEMX_MODEL_CARD.md) are included locally. Its model card links to the separate NVIDIA Open Model License above; the Apache code license is not a substitute for those model terms. The repository-level model license does not erase the separate SAM/MHR/DINOv3 terms. The GEM-X temporal model,
ViTPose, RF-DETR, SOMA-X, and GVHMR are no longer part of this runtime.

## Optional TensorRT acceleration

- NVIDIA TensorRT package: `tensorrt-cu12==11.2.1.2`
- TensorRT license: proprietary NVIDIA Software License Agreement, included in
  the installed wheel; [local copy](../third_party/licenses/TENSORRT_LICENSE.txt)
- NVIDIA Model Optimizer package: `nvidia-modelopt==0.46.0`
- Model Optimizer license: Apache-2.0

These packages are installed only through the `trt` uv extra. TensorRT is not
relicensed by this repository's AGPL-3.0-only license; users must review and comply
with the NVIDIA agreement bundled in the package. Locally built engine files
are machine-, TensorRT-, model-, and precision-specific and are not distributed
with the source repository.

## BiRefNet

- Upstream source and model: <https://huggingface.co/ZhengPeng7/BiRefNet>
- Pinned model revision: `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`
- Upstream code: <https://github.com/ZhengPeng7/BiRefNet>
- License: MIT, copied at `third_party/licenses/BIREFNET_LICENSE`

The adaptive preprocessing path downloads the pinned BiRefNet configuration, inference source, and weights on demand. It uses the resulting foreground masks for source cropping and source-framing analysis. Those downloaded files retain their upstream terms.

## Pexels example media

- Source platform: <https://www.pexels.com/>
- Location: `data/source/pexels`
- Terms: <https://www.pexels.com/license/>

The pinned example list in `fdanyone/assets.py` contains 20 Pexels clips; pipeline inference selects 121 usable frames from an input. Numeric video IDs in the filenames identify the source pages at `https://www.pexels.com/video/<ID>/`. The media remains subject to Pexels terms and is not relicensed under AGPL-3.0.

## DiffSynth-Studio inference runtime

- Public source: <https://github.com/modelscope/DiffSynth-Studio>
- Public base revision: `04e39f7de53df7276a7b40ca1791c2a393e05ff3`
- Original experiment fork revision: `c00782d90c872c97bda4745a9e6a41a0a4a7c4db`
- Location: `fdanyone/vendor/diffsynth`
- License: Apache-2.0, copied at `fdanyone/vendor/diffsynth/LICENSE`

`UPSTREAM.md`, `UPSTREAM.patch`, and `VENDORED_FILES.txt` in that directory record the exact provenance, research patch, and retained file set.

One retained DiffSynth file carries additional code-level upstream attribution:

- `models/wan_video_pose_encoder.py` derives its pose network from [Tencent/MimicMotion](https://github.com/Tencent/MimicMotion/tree/c053153a1d124abae8c08568925ae88debc63001), Copyright Tencent, under Apache-2.0.

## Sapiens2-derived Goliath/MHR schema material

- Source project: <https://github.com/facebookresearch/sapiens2>
- Source revision: `0e51c12d7c7257d88431b2d50e523a7b03004854`
- Source file: `sapiens/pose/configs/_base_/keypoints308.py`
- Retained material: the exact MHR70 name/order and Goliath40 link closure needed to reproduce model conditioning
- Original upstream Apache-2.0 material: RGB palette, color assignment policy, and renderer implementation in `fdanyone/skeleton`
- License copy: `third_party/licenses/SAPIENS2_LICENSE.md`

No Sapiens2 model weights are distributed by this repository.

The retained schema material is conservatively treated as Sapiens2-derived and the complete agreement is included rather than assuming that the root AGPL-3.0-only license applies. The agreement contains explicit prohibited-use terms, including surveillance, biometric processing, identification or re-identification, deepfakes or deceptive content, and the other activities listed in section 1(b)(vi).

## Model and body assets

The 4DAnyone checkpoint is an upstream release artifact licensed under
Apache-2.0 and published at the immutable Hugging Face revision frozen in
`fdanyone/assets.py`. SAM 3D Body and MHR retain the separate terms above.

The frozen base-model assets come from the official [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) repository at revision `37685f96025fc1425edccdd4b2bca3836ae917ff` and the official [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) repository at revision `3f40b6dc4ca5c02dd23c9db74d9d2ccb82903b86`. The latter is used offline to produce the fixed prompt-conditioning tensor; its T5 weights and tokenizer are not part of reader downloads. Their official model cards license those model assets under Apache-2.0. The redistributed copies are pinned by the frozen Hugging Face revision.

## Default Wan2.2 TI2V 5B Turbo LoRA

The rank-64 adapter is mirrored by `AntResearch/4DAnyone` at revision
`4c80e87b805a5f8461cf339cdbe2fb4249e585aa`, with SHA256
`0ace5244e3d1256f884662c261b017249796cf5b95f05d5ed93cc02a478967b8`.
Upstream identifies it as a byte-for-byte mirror from `Kijai/WanVideo_comfy`
revision `86c2b0442e01eeee630b48fd7efc0cd37af03252`, derived from
`quanhaol/Wan2.2-TI2V-5B-Turbo`. Its license is
[CC BY-NC-SA 4.0](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo/blob/77768551236ad110d68299cf1a7151c0c728bddb/LICENSE.md), with a [local copy](../third_party/licenses/TURBO_LORA_LICENSE.md). This adapter is enabled by default, so the full runtime should not be described as entirely AGPL-3.0 or unrestricted for commercial use. Choosing Base generation omits the adapter but does not remove other components' terms.

## SageAttention

The default L40S inference runtime uses SageAttention 2.2.0, source commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, from https://github.com/thu-ml/SageAttention under Apache-2.0. The bundled wheel includes its license ([local copy](../third_party/licenses/SAGEATTENTION_LICENSE)); build target and SHA-256 are recorded in [wheels/README.md](../wheels/README.md).

## Interactive UI

The browser scene renderer and FastAPI adapters are original code in this adaptation under AGPL-3.0-only. The header links to the project's source repository. Three.js 0.186.1 and its OrbitControls use MIT terms; the [license](../third_party/licenses/THREE_LICENSE), source URL and checksum are bundled. Its npm tarball integrity is pinned in `fdanyone/space/web_assets.py`. ColmapView was used as a visual/interaction reference; no ColmapView source code or assets were copied. FastAPI and Uvicorn retain their distributed license texts. See [UI provenance](../fdanyone/space/UPSTREAM.md) and the dependency inventory.

The UI bundles the Latin variable IBM Plex Sans font from `@fontsource-variable/ibm-plex-sans` 5.3.0 under the SIL Open Font License 1.1. Its [license](../third_party/licenses/IBM_PLEX_SANS_LICENSE), source URL and checksum are included. It is served locally, without a browser font CDN.
