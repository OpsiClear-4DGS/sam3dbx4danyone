import * as THREE from 'three';

// One full-resolution MHR animation. Frame changes reuse the same GPU buffers.
export class BodyMesh extends THREE.Mesh {
  static async load(url, frames, signal) {
    const response = await fetch(url, {signal});
    if (!response.ok) throw Error('Body mesh could not be loaded.');
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength < 16) throw Error('Incomplete body mesh.');
    const header = new DataView(buffer);
    const count = header.getUint32(4, true), vertices = header.getUint32(8, true), faces = header.getUint32(12, true);
    if (header.getUint32(0, true) !== 0x3152484d || count !== frames || vertices < 3 || vertices > 100000
        || !faces || faces > 200000 || buffer.byteLength !== 16 + count * vertices * 12 + faces * 12) {
      throw Error('Invalid body mesh animation.');
    }
    const positions = new Float32Array(buffer, 16, count * vertices * 3);
    const indices = new Uint32Array(buffer, 16 + positions.byteLength, faces * 3);
    if (positions.some(v => !Number.isFinite(v)) || indices.some(i => i >= vertices)) throw Error('Invalid body mesh geometry.');
    return new BodyMesh(positions, indices, count, vertices);
  }

  constructor(positions, indices, frames, vertices) {
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3).setUsage(THREE.DynamicDrawUsage));
    super(geometry, new THREE.MeshStandardMaterial({color: 0xaab6c4, roughness: .72, metalness: .05, side: THREE.DoubleSide}));
    this.positions = positions; this.frames = frames; this.vertices = vertices; this.frame = -1;
    this.frustumCulled = false;
    this.setFrame(0);
    this.geometry.attributes.normal.setUsage(THREE.DynamicDrawUsage);
  }

  setFrame(frame) {
    frame = Math.max(0, Math.min(this.frames - 1, Math.floor(frame)));
    if (frame === this.frame) return;
    const size = this.vertices * 3;
    this.geometry.attributes.position.array.set(this.positions.subarray(frame * size, (frame + 1) * size));
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.computeVertexNormals();
    this.frame = frame;
  }
}
