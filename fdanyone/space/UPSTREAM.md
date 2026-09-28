# UI provenance

The Three.js scene integration, web controls, FastAPI queue adapter and calibrated result reader are maintained in this repository under AGPL-3.0-only. The renderer replaces the previous Rerun integration. The UI header links to the corresponding source repository.

Gaussian playback reuses the parser and animated depth sorter from the pinned
FreeTimeGsVanilla submodule (`e782417a574fd9a5a81634bf4a92742e91107a8e`).
`static/gaussian-shaders.js` adapts its `player/renderer.js` shaders for Three.js
attributes, GLSL setup and depth in the shared scene. This adaptation retains
its AGPL-3.0 license and source attribution. The parser and sorter remain unmodified;
its [LICENSE](../../third_party/FreeTimeGsVanilla/LICENSE) and original README
retain the upstream notices.
Its TSOG decoder and audio clock are reused directly, including the
[TSOG/PlayCanvas notices](../../third_party/FreeTimeGsVanilla/player/THIRD_PARTY.md).

https://colmapview.github.io/latest/ (v0.14.5, reviewed 2026-09-27) was used as a visual and interaction reference: dark grid, compact scene controls, camera selection and an adjacent image pane. No ColmapView code or assets were copied.

Three.js 0.186.1 supplies WebGL rendering, video textures and OrbitControls. The official npm tarball URL and SHA-512 integrity are pinned in `web_assets.py`. Only the required modules, package metadata and MIT license are extracted to the external UI cache and served locally. Its license is also bundled in `third_party/licenses/THREE_LICENSE`.

Camera transforms use the existing canonical-human-world / OpenCV-camera contracts without conversion of saved outputs. SAM keypoint topology is provided by `fdanyone.skeleton.keypoints`, whose source and licensing remain documented there.

The animated body surface and triangle topology come from the existing SAM 3D Body / MHR assets. The viewer adds no model weights or external geometry assets. Their license notices remain in [SAM3D_BODY_LICENSE](../../third_party/licenses/SAM3D_BODY_LICENSE), [MHR_LICENSE](../../third_party/licenses/MHR_LICENSE) and the repository's [third-party notices](../../docs/THIRD_PARTY_NOTICES.md).

The neutral grayscale workbench, compact icon rail and edge-to-edge scene follow ColmapView’s visual style. IBM Plex Sans is bundled from the official Fontsource npm package `@fontsource-variable/ibm-plex-sans` 5.3.0, under SIL OFL 1.1, with its license in `third_party/licenses/IBM_PLEX_SANS_LICENSE`. Icons and CSS are implemented locally.
