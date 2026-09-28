"""Preserve a source audio track on the generated clip's presentation clock."""
import math
import os
from pathlib import Path
import subprocess
import tempfile

import av


def extract_audio(source, destination, *, start, end, duration):
    """Trim absolute source PTS and change tempo to fit the selected video span.

    The first audio stream is retained as AAC in M4A. Timestamp gaps are padded
    with silence; a video without audio returns None. No generated camera audio
    is synthesized. ``end`` is exclusive and ``duration`` is output seconds.
    """
    if not all(math.isfinite(v) for v in (start, end, duration)) or end <= start or duration <= 0:
        raise ValueError('Audio trim needs finite ordered times and a positive duration.')
    source, destination = Path(source), Path(destination)
    with av.open(str(source)) as container:
        if not container.streams.audio:
            return None
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    speed = (end-start)/duration
    tempos = []
    while speed > 2:
        tempos.append('atempo=2'); speed /= 2
    while speed < .5:
        tempos.append('atempo=0.5'); speed *= 2
    tempos.append(f'atempo={speed:.12g}')
    filters = [f'asetpts=PTS-({start:.12g})/TB', 'aresample=async=1:first_pts=0',
               f'atrim=duration={end-start:.12g}', 'asetpts=PTS-STARTPTS', *tempos,
               'apad', f'atrim=duration={duration:.12g}']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.audio-', dir=destination.parent) as temporary:
        staged = Path(temporary)/'track.m4a'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-copyts',
                        '-i', str(source), '-map', '0:a:0', '-vn', '-af', ','.join(filters),
                        '-t', str(duration), '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                        '-ac', '2', '-movflags', '+faststart', str(staged)], check=True)
        os.link(staged, destination)
    return destination


def export_clip_audio(clip, destination):
    """Publish audio for a canonical clip, including prepared silent chunks."""
    source = clip.source_path
    prepared = source.with_suffix('.audio.m4a')
    if prepared.is_file():
        # Prepared chunk audio already follows its frame skipping and trim.
        if clip.frames[0].source_timestamp == 0 and clip.input_rate == clip.fps:
            from fdanyone.io import link_or_copy_file
            return link_or_copy_file(prepared, destination)
        source = prepared
    start = float(clip.frames[0].source_timestamp)
    end = float(clip.frames[-1].source_timestamp)
    duration = (len(clip.frames)-1)/float(clip.fps)
    return extract_audio(source, destination, start=start, end=end, duration=duration)
