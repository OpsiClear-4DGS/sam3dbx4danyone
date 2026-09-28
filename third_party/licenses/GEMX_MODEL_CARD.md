---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
tags:
  - pose-estimation
  - human-motion
  - soma-body-model
  - smpl-body-model
  - video
  - monocular-video
  - 3d-pose
  - transformer
library_name: gem
---


# Model Overview

### Description:
**Ge**neralist **M**odel for Human Motion (GEM) is a monocular video 3D human body pose estimation model developed by NVIDIA. GEM reconstructs full-body motion — including body and hands — from unconstrained video sequences with dynamic cameras, producing accurate per-frame 3D body pose and world-space global motion trajectories in SOMA format. The model outputs 77-joint poses using the SOMA parametric body model, recovering both local body kinematics and global motion trajectories from monocular video.

All training data and the SOMA parametric body model are owned by NVIDIA or released under permissive licenses, making GEM ready for commercial use.

## License/Terms of Use:
Use of the source code is governed by the [Apache License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md), Version 2.0. Use of the associated model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). 

## Deployment Geography:
Global

## Use Case:
GEM is intended for use by individuals and professionals in fields such as machine learning and computer vision research, gaming, animation, 3D content creation, and biomechanics. Specific use cases include:
- **Motion capture from video** — reconstruct accurate 3D body pose and global motion trajectories from monocular video with dynamic cameras, useful for recovering full-body motion in unconstrained environments.
- **Character animation** — create realistic full-body character animations derived from real video footage for gaming and film applications.
- **Digital humans** — drive SOMA-compatible avatar rigs directly from video input.
- **Biomechanical analysis** — study human movement for applications in sports, rehabilitation, and medicine.

## Expected Release Date:
GitHub: 03/16/2026 <br>
Hugging Face: 03/16/2026

## Reference(s):
- GENMO: A GENeralist Model for Human MOtion — Li et al., ICCV 2025 (released as GEM)
- SOMA: Unifying Parametric Human Body Models 
- Project Page: https://research.nvidia.com/labs/dair/gem/
- Paper: https://arxiv.org/abs/2505.01425

## Model Architecture:
**Architecture Type:** Transformer (Regression-Based) <br>
**Network Architecture:** GEM uses a 12-layer RoPE-based Transformer encoder for temporal motion modeling. The forward pass consists of three stages:
1. **Additive Fusion Block** — per-frame video features, person bounding box, and camera intrinsics are independently processed by dedicated MLPs and summed to form a unified conditioning token per frame.
2. **Transformer Encoder (×12 layers)** — each layer comprises LayerNorm, a RoPE attention layer with residual connections, and an MLP. Attention units have 8 heads; the MLP hidden dimension is d_mlp = 512. Rotary Position Embeddings (RoPE) enable variable-length sequence processing and a sliding-window attention mechanism at inference for sequences longer than those seen during training.
3. **Regression Head** — a linear projection decodes the latent sequence into per-frame 585-dim SOMA feature vectors, which are decoded to full SOMA body parameters by the EnDecoder.

The model is a pure regression model — there are no diffusion or iterative sampling steps at inference. <br>

**This model was developed independently by NVIDIA.** <br>

**Number of model parameters:** 5.2 × 10⁸

## Computational Load:
**Training compute:** 500,000 steps, batch size 128 distributed across 8 NVIDIA A100 80GB GPUs, AdamW optimizer (lr = 2 × 10⁻⁴), 16-bit mixed precision, gradient clipping at 0.5. Cumulative compute: ~1.9 × 10¹⁹ FLOPs. Training energy: ~57.6 kWh (8 GPUs × 100W average × 72 hours).<br>
**Inference:** Single feed-forward pass per video clip; sliding-window attention (window size W = 120) for arbitrary-length video at inference time.

## Input(s):
**Input Type(s):** Video (RGB), person bounding box, camera intrinsics <br>

**Input Format(s):**
- Video frames: RGB tensor `(T, H, W, 3)`, any resolution (internally processed via frozen video encoder)
- Person bounding box: `(T, 4)` float array in xyxy pixel coordinates
- Camera intrinsics: `(T, 3)` float array in Cliff representation (focal length + principal point offsets)

**Input Parameters:** Three-Dimensional (3D) video tensor; One-Dimensional (1D) bounding box and intrinsics per frame <br>

**Other Properties Related to Input:**
All three inputs are estimated automatically from the input video during inference — no manual annotation is required. Bounding boxes are extracted via SAM-3D-Body tracker; and camera intrinsics are estimated via focal-length estimation. The model processes training sequences of length N = 120 frames; longer video sequences are handled at inference via sliding-window attention.

## Output(s):
**Output Type(s):** Human body motion in SOMA format (world-space) <br>

**Output Format(s):** PyTorch float32 tensors <br>

**Output Parameters:** Three-Dimensional (3D) per-frame body parameters <br>

- Per-frame body pose: `(T, 77, 3)` axis-angle joint rotations (SOMA 77-joint skeleton, full body including hands)
- Global root orientation: `(T, 3)` axis-angle in world space
- World-space root translation: `(T, 3)` in meters
- Identity coefficients: `(T, 64)` SOMA identity shape vector (64-component PCA shape space for SOMA-native backend)
- Scale parameters: `(T, 69)` body-part scale parameters
- Internally encoded as a 585-dim SOMA feature vector per frame

**Other Properties Related to Output:**
Output is decoded from the 585-dim SOMA feature space to per-frame SOMA body parameters. Root translation is recovered in world (global) coordinates using the estimated camera trajectory. Global scale is clamped to [0.7, 1.0] during decoding for numerical stability.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (GPU cores) and software frameworks (CUDA libraries), the model achieves faster inference compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):**
- PyTorch 2.10.0
- PyTorch Lightning 2.6.1

**Supported Hardware Microarchitecture Compatibility:**
- NVIDIA Ampere (A100 recommended) — tested on A100 80GB
- NVIDIA Hopper (H100)
- NVIDIA Ada Lovelace (RTX 4000-series, L40)
- NVIDIA Blackwell

**Preferred/Supported Operating System(s):**
- Linux

**Required:** CUDA 12.1+

## Model Version(s):
- **GEM v1.0 (SOMA)** — initial public release; video-to-motion estimation producing 77-joint full-body pose (body + hands) in SOMA format.

## Training and Testing Datasets:

### Training Dataset:

GEM is trained exclusively on internally generated synthetic video data. The training corpus is constructed by combining four NVIDIA-owned and commercially licensed asset sources to render synthetic video sequences with ground-truth SOMA body annotations:

1. **Bones RigPlay-1**  — 350,000 motion capture animation sequences owned by NVIDIA, providing diverse human motion dynamics.
2. **Bones Audio2Gesture dataset - 447 motion capture animation sequences owned by NVIDIA. 
2. **RenderPeople**  — 500 high-fidelity 3D digital human characters owned by NVIDIA, providing photorealistic body appearances.
3. **Internal Synthetic Characters**  — 3,500 internally developed synthetic 3D characters developed by NVIDIA.
4. **HDRI Haven** — 448, 4K HDRI environment maps downloaded from HDRI Haven under a free commercial use license, used as scene backgrounds.

Synthetic training videos are rendered by placing multiple characters in the same scene with randomized camera movements. Per-frame ground-truth SOMA body parameters are derived directly from the mocap sequences, parameterized using the SOMA rigged DH body model. No real-world sensor data or footage of real individuals is used for training.


**Total Number of Asset Sources:** 5 <br>
**Dataset Partition:** Training ~90%, Validation ~10% <br>
**Data Collection Method:** Hybrid — Automatic (synthetic rendering pipeline combining mocap animations, 3D character meshes, and HDRI environments) <br>
**Labeling Method:** Automated — ground-truth annotations are derived directly from the synthetic rendering pipeline (no manual labeling) <br>
**Data Modality:** Video <br>
**Video Training Data Size:** 100k videos <br>
**Dataset License(s):**
- RenderPeople (NSPECT-6W6X-C1GY) NVIDIA purchased dataset of 3D human characters. 
- HDRI Haven (NSPECT-7LPR-0B4G) is a collection of 448 HDRI images from HDRI Haven.  https://jirasw.nvidia.com/browse/DGPTT-4820    
- Bones Rigplay (NSPECT-DQGK-BD6I): NVIDIA purchased dataset with full license to train and distribute AI models. 
- Bones Audio2Gesture dataset (NSPECT-LL4C-7UWJ): NVIDIA purchased  dataset with full license to train AI models on it. 
- NVIDIA Digital Human Characters (NSPECT-TT77-12RM): A collection of 3500 digital human characters synthetically generated using NVIDIA's in-house tool: https://dhgen.ngc.nvidia.com/ 

### Testing Dataset:
Internal validation split drawn from the same synthetic rendering pipeline as the training data (held-out subset).

**Data Collection Method:** Automatic (synthetic rendering) <br>
**Labeling Method:** Automated (synthetic ground truth) <br>
**Dataset License(s):** Internal NVIDIA

## Inference:
**Test Hardware:** NVIDIA A100 80GB (primary benchmark hardware) <br>

**Demo:**
```bash
python scripts/demo/demo_soma.py --video <video.mp4> --ckpt_path inputs/pretrained/gem_soma.ckpt
```

**Pretrained checkpoint:** https://registry.ngc.nvidia.com/orgs/nvidian/models/gem

## Evaluation Dataset 
5,000 synthetically generated videos.   

## Performance Metrics:
Evaluated on our internal synthetic data using world-space mean per-joint position error (W-MPJPE, SOMA 77-joint skeleton): 

| Dataset | Metric | GEM (SOMA) |
|:--------|:-------|:-----------|
| Internal | W-MPJPE (mm) ↓ | 115.2 |

## Disclaimer 
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

## Ethical Considerations:
GEM processes video to estimate human body pose and motion. Users are responsible for ensuring they have proper rights and permissions for all input video content. If input video includes real people, users are responsible for obtaining appropriate consent and complying with applicable privacy regulations. The pose estimates produced by GEM do not contain facial identity or texture information, but body motion patterns may still constitute sensitive data in certain jurisdictions. 

GEM is designed to estimate body pose for legitimate research, creative, and analytical applications. Misuse for non-consensual surveillance, deepfake generation, or privacy violations is contrary to the intended use and may violate applicable law. Users integrating GEM into applications involving real people are responsible for obtaining appropriate consent, implementing privacy safeguards (e.g., anonymizing output before storage, limiting data retention), and complying with applicable regulations. NVIDIA encourages developers to implement content guardrails, access controls, and audit logging in any user-facing application built on top of GEM.

Users are responsible for model inputs and outputs, including implementing guardrails and other safety mechanisms prior to deployment.

For more detailed information on ethical considerations for this model, please see the Model Card++ subcards: [BIAS.md](./docs/BIAS.md), [EXPLAINABILITY.md](./docs/EXPLAINABILITY.md), [SAFETY_and_SECURITY.md](./docs/SAFETY_and_SECURITY.md), and [PRIVACY.md](./docs/PRIVACY.md).

Please report model quality, risk, or security vulnerabilities [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


## Bias

Field | Response
:-----|:--------
Participation considerations from adversely impacted groups ([protected classes](https://www.senate.ca.gov/content/protected-classes)) in model design and testing: | GEM is trained entirely on synthetically rendered video featuring synthetic 3D characters. No real individuals were involved in data collection. The 500 RenderPeople characters and 3,500 internally developed synthetic characters used to populate training scenes represent a designed set of body shapes and skin appearances. The demographic diversity of these character assets — including skin tone, body proportions, age group, and sex — is limited by the character library and may not proportionally represent all global populations. In particular, body proportions and appearances common in East Asian, South Asian, Sub-Saharan African, and other underrepresented groups may be less prevalent in the synthetic training distribution. No adversely impacted groups were formally consulted during model design or testing.
Measures taken to mitigate against unwanted bias: | (1) **Motion diversity independent of appearance**: Bones RigPlay-1 provides 350,000 motion capture sequences spanning a wide variety of activities, movement styles, and body dynamics. Pose estimation accuracy is thus not limited to the appearance diversity of the character asset library. (2) **SOMA body model decoupling**: The output representation (77-joint SOMA skeleton) is identity-agnostic — body shape and appearance at inference time need not match the synthetic training characters; the model generalizes to real-world body proportions via the SOMA parametric representation. (3) **Avoidance of real-video demographic bias**: Unlike models trained on internet-scraped video, GEM's training data does not inherit demographic sampling biases from real-world video collection pipelines (e.g., geographic or socioeconomic biases in camera availability). (4) **Multiple simultaneous subjects**: Training scenes contain multiple synthetic characters per video, helping the model learn pose estimation independently of specific character-level appearance correlations.
Bias Metric (If Measured): | No formal demographic bias metric has been measured for GEM across demographic subgroups.


## Explainability

Field | Response
:-----|:--------
Intended Task/Domain: | 3D Human Body Pose Estimation from monocular video — recovering full-body pose (77 joints, body + hands) and world-space global motion trajectories from unconstrained video with dynamic cameras.
Model Type: | Regression-based Transformer. Pure feed-forward architecture; no iterative sampling or diffusion steps at inference.
Intended Users: | Computer vision researchers; graphics and animation engineers; machine learning engineers; game developers; biomechanics and sports science, and robotics researchers.
Output: | Per-frame SOMA body parameters: 77-joint axis-angle pose `(T, 77, 3)`, world-space root translation `(T, 3)` in meters, identity coefficients, and scale parameters. Internally encoded as a SOMA feature vector per frame and decoded by the SOMA body model to 3D joint positions and mesh vertices.
Describe how the model works: | GEM processes a monocular video sequence through four sequential stages: (1) **Pre-processing** — A person detector and tracker localizes the subject per frame and produces bounding boxes; camera intrinsics are estimated from the video. (2) **Feature extraction** — A frozen video encoder (SAM-3D-Body) extracts per-frame appearance features from crops around the bounding box. (3) **Regression Transformer** — A 12-layer RoPE-based Transformer takes per-frame video features, bounding box coordinates, and camera intrinsics as input via an additive fusion block. Each layer applies LayerNorm, multi-head self-attention with RoPE positional embeddings (8 heads, d_mlp = 512), and a residual MLP. The Transformer outputs a 585-dim SOMA feature vector per frame via a linear regression head. (4) **Decoding** — The SOMA EnDecoder maps the feature vector to interpretable body parameters (body pose, global orientation, world-space translation, identity coefficients, scale), and the SOMA body model performs forward kinematics to produce 3D joint positions and mesh vertices.
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Not formally tested across demographic subgroups. Evaluation on the held-out test set shows accurate pose estimation, but no per-group demographic breakdown was performed.
Technical Limitations & Mitigation: | (1) **Camera trajectory dependency** — world-space global motion recovery requires an accurate camera trajectory from Simultaneous Localization and Mapping (SLAM); SLAM errors (from fast camera motion, textureless environments, or motion blur) propagate directly to world-space translation errors. Mitigation: the model supports a static-camera mode (`--static_cam`) that bypasses SLAM. (2) **Single-person per forward pass** — the model processes one person per inference call (bounding-box cropped). Multi-person scenes require running the model once per tracked subject. (3) **Synthetic-to-real domain gap** — the model is trained exclusively on synthetic video; performance may degrade on real-world appearances, lighting conditions, or poses outside the synthetic training distribution. (4) **Body-only output (no face)** — the model estimates 77-joint full-body pose, including hands but does not estimate facial expressions. (5) **Fixed training context length** — trained on sequences of N = 120 frames; longer sequences use sliding-window attention at inference, which may introduce temporal boundary effects. (6) **Regression without uncertainty** — the model produces point estimates without per-frame confidence scores; users cannot directly detect failures from the output. (7) **Optional module availability** — if the SAM-3D-Body tracker or camera trajectories are unavailable at inference time, the demo pipeline gracefully falls back to full-frame bounding boxes and a static camera trajectory, respectively, allowing the model to still produce local pose estimates but without reliable world-space translation.
Verified to have met prescribed NVIDIA quality standards: | Yes.
Performance Metrics: | W-MPJPE (world-space mean per-joint position error, 77 joints) = **115.2 mm** on the held-out test set.
Potential Known Risks: | (1) **Privacy** — GEM can reconstruct detailed body motion from video of real individuals. Users must ensure appropriate consent and legal compliance before applying the model to video of real people. (2) **Non-consensual motion capture** — recovered body motion could be used to animate avatar rigs in misleading or non-consensual contexts (e.g., deepfakes). (3) **Silent failure on out-of-distribution inputs** — unusual clothing, heavy occlusion, or non-standard body proportions may produce inaccurate estimates without explicit warning signals in the output.
Licensing: | Use of the source code is governed by the [Apache License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md), Version 2.0. Use of the associated model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).


## Privacy

Field | Response
:-----|:--------
Generatable or reverse engineerable personal data? | Partial — GEM generates 3D body pose estimates from input video. If the input video contains real individuals, the output SOMA body parameters represent an anonymized geometric reconstruction of their body motion; no texture, facial geometry, or biometric identity is encoded in the SOMA output.
Personal data used to create this model? | Partial — GEM's training video is synthetically rendered and contains no real-person footage. However, Bones RigPlay-1 is a motion capture dataset of 350,000 animations that are recorded from real human performers. However, the data was retargeted to fixed skeleton removing any person specific biometric signals. All other training components (RenderPeople, internal synthetic characters, HDRI Haven) are fully synthetic or non-personal assets.
Was consent obtained for any personal data used? | Yes
Description of methods implemented in data acquisition or processing, if any, to address the prevalence of personal data in the training data: | Training video is synthetically rendered — no real individuals appear visually in the training footage. The rendering pipeline combines motion capture animations (Bones RigPlay-1), 3D character meshes (RenderPeople, internal synthetic characters), and HDRI environment maps (HDRI Haven) to produce fully synthetic video. Bones RigPlay-1 motion sequences are used only as skeletal animation data; no visual appearance, face, texture, or identity information from any real performers is included in training data or model outputs. Ground-truth SOMA body parameter annotations are derived automatically from the synthetic rendering pipeline.
How often is dataset reviewed? | Datasets are initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for changes.
Is a mechanism in place to honor data subject right of access or deletion of personal data? | Not Applicable — training data contains no real-person data. No data subject rights apply to the synthetic training corpus. For the RenderPeople and internal character assets, access is governed by the respective  third-party collectors.
If personal data was collected for the development of the model, was it collected directly by NVIDIA? | No 
If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects? | Not Applicable (Externally-Sourced Data) 
If personal data was collected for the development of this AI model, was it minimized to only what was required? | Yes.
Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? | No
Is there provenance for all datasets used in training? | Yes 
Does data labeling (annotation, metadata) comply with privacy laws? | Yes — all annotations are automatically generated from the synthetic rendering pipeline; no personal data is annotated.
Is data compliant with data subject requests for data correction or removal, if such a request was made? | No, not possible for externally sourced data. 
Applicable Privacy Policy | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/


## Safety

Field | Response
:-----|:--------
Model Application Field(s): | Media & Entertainment; Gaming and Animation; Computer Vision Research; Biomechanics and Sports Science; 3D Content Creation, Robotics
Describe the life critical impact (if present). | Not Applicable — GEM is a body pose estimation model. It should not operate in safety-critical control loops (autonomous vehicles, medical devices, industrial safety systems). While it may be used as a component in biomechanics or rehabilitation research pipelines, the model itself does not make safety-critical decisions.
Use Case Restrictions: | Abide by the [NVIDIA License](https://gitlab-master.nvidia.com/dair/projects/gem/-/blob/main/LICENSE?ref_type=heads). GEM must not be used to: (1) process video of individuals without appropriate legal authorization and, where required, explicit consent; (2) create non-consensual synthetic representations, deepfakes, or synthetic identity impersonations of real individuals; (3) conduct unauthorized surveillance or tracking of individuals; (4) violate applicable privacy, biometric, or data protection laws in the deployment jurisdiction (e.g., GDPR, CCPA, BIPA). Integration into safety-critical systems (medical devices, autonomous vehicles, industrial machinery) requires additional validation by the integrating team.
Model and dataset restrictions: | The Principle of Least Privilege (PoLP) is applied, limiting access for dataset generation and model development. The training corpus (Bones RigPlay-1, RenderPeople, internal synthetic characters, HDRI Haven) contains no real-person video or personally identifiable information; all subjects in training data are synthetic 3D characters. NSpect IDs are maintained for all training data sources.
Security considerations: | GEM accepts RGB video and float arrays (bounding boxes, camera intrinsics) as inputs. It does not accept executable inputs and makes no network calls at inference time. The pre-processing pipeline relies on third-party libraries (SAM-3D-Body); users should verify package integrity via official distribution channels. Input videos should be validated for appropriate content and provenance before processing. The model checkpoint should be obtained from the official NVIDIA distribution channel and its integrity verified. Report security vulnerabilities to NVIDIA [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
Responsible AI practices: | GEM is designed to estimate body pose for legitimate research, creative, and analytical applications. Misuse for non-consensual surveillance, deepfake generation, or privacy violations is contrary to the intended use and may violate applicable law. Users integrating GEM into applications involving real people are responsible for obtaining appropriate consent, implementing privacy safeguards (e.g., anonymizing output before storage, limiting data retention), and complying with applicable regulations. NVIDIA encourages developers to implement content guardrails, access controls, and audit logging in any user-facing application built on top of GEM.
