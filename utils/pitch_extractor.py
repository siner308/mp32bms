# https://github.com/librosa/librosa/issues/1102#issuecomment-627454351

import librosa
import numpy as np
from math import tau


def freq_power_spectrum(y, n_fft=1024, hop_length=256, center=True, pad_mode='reflect', **stft_args):
    if center:
        y = np.pad(y, (n_fft + hop_length) // 2, mode=pad_mode)
    norm = 2 ** .5 / .612 / n_fft  # This ought to be handled within stft
    Y = librosa.stft(y * norm, n_fft=n_fft, hop_length=hop_length, center=False, **stft_args)
    power = abs(Y) ** 2
    # Use phase differences to calculate true frequencies
    dp = np.diff(np.angle(Y) / tau, axis=-1)
    rp = np.fft.rfftfreq(n_fft)[..., None].astype(np.float32)
    # Return dimensions (freq|power) x bin x frame
    return np.stack([
        (np.round(rp * hop_length - dp) + dp) / hop_length,
        0.5 * (power[..., 1:] + power[..., :-1]),
    ])


def group_by_freq(frame, tol=4096 ** -1, min_power=1e-7):
    """Group similar frequencies into power-weighted mean freq and total power."""
    frame = frame[..., (frame[0] > 0) & (frame[1] > min_power)]
    # Sort by frequency
    frame = frame[..., np.argsort(frame[0])]
    # Combine similar peaks together by power-weighted average
    idx, = np.nonzero(np.diff(frame[0], prepend=0) > tol)
    p = np.add.reduceat(frame[1], idx)
    f = np.add.reduceat(frame[1] * frame[0], idx) / p
    frame = np.stack([f, p])
    # Sort by power (strongest first)
    frame = frame[..., np.argsort(-frame[1])]
    return frame


def match_harmonics(freqs, powers=None, max_tones=4, min_power=1e-5):
    """Combine harmonics into tones."""
    tones = []
    if freqs.shape:
        if powers is None:
            powers = np.ones(freqs.shape)
        m = np.divide.outer(freqs, freqs)
        harm = m.round()
        harm[abs(m - harm) > 0.1] = 0
        m = powers[:, None] * (harm > 0)
        # Construct tones using peaks matched by the winning hypothesis
        while len(tones) < max_tones and m.any():
            winner = np.argmax(m.sum(axis=0))
            matched, = np.where(m[:, winner])
            tone = np.average(
                freqs[matched] / harm[matched, winner],
                weights=powers[matched],
                returned=True
            )
            if tone[1] < min_power:
                break
            tones.append(tone)
            # Clear the winning hypothesis and any matched peaks
            m[:, winner] = 0
            m[matched, :] = 0
    return np.array(tones, dtype=np.float32).reshape(-1, 2)  # Reshape in case of empty


def multipitch(y, n_fft=1024, hop_length=256, center=True, max_tones=4, min_power=1e-5):
    """Multitonal f0 detection."""
    spectr = freq_power_spectrum(y, n_fft, hop_length, center)
    return [
        match_harmonics(
            *group_by_freq(frame, tol=1.0 / n_fft)[:, :32],
            max_tones=max_tones,
            min_power=min_power,
        )
        for frame in spectr.transpose(2, 0, 1)
    ]


def bestpitch(y, n_fft=1024, hop_length=256, center=True, sr=1.0, min_power=1e-5):
    """Estimate the dominating fundamental frequency for y over time."""
    ret = multipitch(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        center=center,
        max_tones=1,
        min_power=min_power,
    )
    return np.array([
        sr * r[0, 0] if r.size and r[0, 1] > min_power else 0.0
        for r in ret
    ])
