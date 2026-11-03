#!/usr/bin/env python3
"""Encode raw RGB frames from stdin, then fully decode/verify the new MP4."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--expected-frames', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite encoded output')
    count = 0
    size = args.width * args.height * 3
    with av.open(str(args.output), mode='w') as container:
        stream = container.add_stream('libx264', rate=args.fps)
        stream.width, stream.height, stream.pix_fmt = args.width, args.height, 'yuv420p'
        stream.codec_context.thread_count = 1
        stream.options = {'crf': '18', 'preset': 'veryfast', 'threads': '1'}
        while True:
            payload = sys.stdin.buffer.read(size)
            if not payload:
                break
            if len(payload) != size:
                raise ValueError('Truncated RGB frame from renderer')
            pixels = np.frombuffer(payload, np.uint8).reshape(args.height, args.width, 3)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = count, Fraction(1, args.fps)
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    if count != args.expected_frames:
        raise ValueError(f'Expected {args.expected_frames} frames, received {count}')
    decoded = 0
    with av.open(str(args.output)) as container:
        video = container.streams.video[0]
        video.codec_context.thread_count = 1
        for frame in container.decode(video):
            if (frame.width, frame.height) != (args.width, args.height):
                raise ValueError('Decoded frame dimensions differ')
            if frame.pts * frame.time_base != Fraction(decoded, args.fps):
                raise ValueError('Decoded video timestamps differ')
            decoded += 1
    if decoded != count:
        raise ValueError('Full decode frame count differs')
    digest = hashlib.sha256()
    with args.output.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    result = {'schema': 'nepes_visual_replay_mp4_v1', 'verified': True,
              'frames': decoded, 'fps': args.fps, 'resolution': [args.width, args.height],
              'sha256': digest.hexdigest(), 'full_decode': True, 'native_dataset_rgb': False}
    with args.output.with_suffix('.verification.json').open('x') as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
