"""Pinned, locally served Three.js modules; no browser CDN dependency."""
import base64
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request

VERSION = '0.186.1'
URL = f'https://registry.npmjs.org/three/-/three-{VERSION}.tgz'
INTEGRITY = 'blFeqb49wRCSGUGj7gtpfnSGHy2lwDk94RhUmS1c/hTby70kvChbWpkJ4Pm1390LqzzvTmzgXKHPEafJwCb8jA=='
FILES = ('build/three.module.js', 'build/three.core.js', 'examples/jsm/controls/OrbitControls.js', 'LICENSE', 'package.json')



def prepare_web_assets(cache_dir):
    root = Path(cache_dir)/f'three-{VERSION}'
    marker = root/'verified.txt'
    if marker.is_file() and marker.read_text() == INTEGRITY and all((root/n).is_file() for n in FILES):
        return root
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    if base64.b64encode(hashlib.sha512(data).digest()).decode() != INTEGRITY:
        raise ValueError('Three.js web package checksum mismatch.')
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for name in FILES:
            member = archive.getmember('package/'+name)
            if not member.isfile():
                raise ValueError('Unexpected Three.js package entry.')
            (root/name).parent.mkdir(parents=True, exist_ok=True)
            temporary = root/(name+'.tmp')
            temporary.write_bytes(archive.extractfile(member).read())
            temporary.replace(root/name)
    marker.write_text(INTEGRITY)
    return root
