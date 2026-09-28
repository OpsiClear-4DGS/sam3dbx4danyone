"""Frame-indexed trim, acceleration and lossless 121-frame chunk preparation."""
from bisect import bisect_left
from contextlib import ExitStack
from fractions import Fraction
import math
from pathlib import Path

import av

from fdanyone.config import INFERENCE
from fdanyone.io import write_json
from fdanyone.video import _frame_timestamp, _stream_rate, _rotation_degrees, _frame_rotation_degrees, _frame_to_rgb


def video_index(path):
    """Cache decoded presentation times, so VFR trims use actual frame boundaries."""
    import json
    path = Path(path)
    stat = path.stat()
    identity = [stat.st_size, stat.st_mtime_ns]
    cache = path.with_suffix(path.suffix + '.index.json')
    if cache.exists():
        data = json.loads(cache.read_text())
        if data['identity'] == identity:
            return data
    timestamps = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('The upload has no video stream.')
        stream = container.streams.video[0]
        fps = _stream_rate(stream)
        for i, frame in enumerate(container.decode(stream)):
            stamp = float(_frame_timestamp(frame, stream, i))
            if not timestamps:
                origin = stamp
            timestamps.append(stamp-origin)
    if not timestamps or any(b < a for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError('Video must contain frames with ordered presentation timestamps.')
    data = dict(identity=identity, timestamps=timestamps, frames=len(timestamps),
                fps_num=fps.numerator, fps_den=fps.denominator,
                duration=timestamps[-1]+float(1/fps))
    write_json(cache, data)
    return data


def plan_clip(path, start=0., end=None, skip=0):
    index = video_index(path)
    end = index['duration'] if end is None else end
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= index['duration'] + 1e-6:
        raise ValueError('Choose a trim start and end within the video, with end after start.')
    if isinstance(skip, bool) or not isinstance(skip, int) or not 0 <= skip <= 15:
        raise ValueError('Skip must be an integer between 0 and 15.')
    times = index['timestamps']
    first, stop = bisect_left(times, start), bisect_left(times, end)
    stride = skip+1
    kept = len(range(first, stop, stride))
    length = INFERENCE.num_frames
    if kept < length:
        raise ValueError(f'This selection keeps {kept} frames; at least {length} are needed. Extend the trim or skip fewer frames.')
    starts = list(range(0, kept-length+1, length))
    if starts[-1]+length < kept:
        starts.append(kept-length)
    chunks = []
    for i, offset in enumerate(starts):
        source_first = first+offset*stride
        source_last = first+(offset+length-1)*stride
        chunks.append(dict(index=i, label=f'Chunk {i+1}', first=source_first, last=source_last,
                           start=times[source_first], end=min(end, times[source_last]+index['fps_den']/index['fps_num']),
                           overlap_frames=max(0, starts[i-1]+length-offset) if i else 0))
    return dict(source=str(Path(path).resolve()), identity=index['identity'], start=start, end=end,
                skip=skip, stride=stride, kept_frames=kept, chunks=chunks,
                fps_num=index['fps_num'], fps_den=index['fps_den'],
                chunk_duration=length*index['fps_den']/index['fps_num'])


def prepare_chunks(plan, directory):
    """Decode once; stream selected RGB frames to at most two lossless encoders."""
    source = Path(plan['source'])
    stat = source.stat()
    if [stat.st_size, stat.st_mtime_ns] != plan['identity']:
        raise ValueError('The source video changed after the trim was planned.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fps = Fraction(plan['fps_num'], plan['fps_den'])
    chunks = plan['chunks']
    outputs = [directory/f'chunk_{i+1:03d}.mp4' for i in range(len(chunks))]
    active = {}
    next_chunk = 0
    with ExitStack() as stack:
        def close_pending():
            for j, (output, _, _) in active.items():
                output.close()
                outputs[j].with_suffix('.part.mp4').unlink(missing_ok=True)
        stack.callback(close_pending)
        container = stack.enter_context(av.open(str(source)))
        has_audio = bool(container.streams.audio)
        stream = container.streams.video[0]
        rotation = _rotation_degrees(stream)
        observed_rotation = None
        for i, frame in enumerate(container.decode(stream)):
            if i == 0:
                origin = float(_frame_timestamp(frame, stream, 0))
            if i > chunks[-1]['last']:
                break
            while next_chunk < len(chunks) and i == chunks[next_chunk]['first']:
                target = outputs[next_chunk]
                output = av.open(str(target.with_suffix('.part.mp4')), mode='w')
                encoder = output.add_stream('libx264rgb', rate=fps)
                encoder.pix_fmt = 'rgb24'
                encoder.options = {'crf':'0', 'preset':'veryfast'}
                active[next_chunk] = [output, encoder, 0]
                next_chunk += 1
            selected = [j for j in active if (i-chunks[j]['first']) % plan['stride'] == 0]
            if not selected:
                continue
            rotation_now = _frame_rotation_degrees(frame, rotation)
            if observed_rotation is not None and rotation_now != observed_rotation:
                raise ValueError('Video rotation changes during the selected clip.')
            observed_rotation = rotation_now
            rgb = _frame_to_rgb(frame, rotation_now)
            for j in selected:
                output, encoder, count = active[j]
                if count == 0:
                    encoder.height, encoder.width = rgb.shape[:2]
                converted = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                converted.pts = count
                converted.time_base = 1/fps
                for packet in encoder.encode(converted):
                    output.mux(packet)
                active[j][2] += 1
                if i == chunks[j]['last']:
                    if active[j][2] != INFERENCE.num_frames:
                        raise ValueError('Chunk did not contain exactly 121 frames.')
                    for packet in encoder.encode():
                        output.mux(packet)
                    output.close()
                    outputs[j].with_suffix('.part.mp4').replace(outputs[j])
                    del active[j]
        if active or next_chunk != len(chunks):
            raise ValueError('Source ended before all planned chunks were written.')
    if has_audio:
        from fdanyone.audio import extract_audio
        times = video_index(source)['timestamps']
        for chunk, target in zip(chunks, outputs, strict=True):
            # A continuous TSOG spans the first through last sampled frame.
            extract_audio(source, target.with_suffix('.audio.m4a'),
                          start=origin+chunk['start'], end=origin+times[chunk['last']],
                          duration=(INFERENCE.num_frames-1)/float(fps))
    write_json(directory/'manifest.json', plan)
    return [str(p) for p in outputs]
