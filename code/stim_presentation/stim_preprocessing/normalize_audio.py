#!/usr/bin/env python3
"""
Audio Normalization Script for Pilot Study
Normalizes all WAV files in the stim folder to a target RMS level.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse


def normalize_audio(input_path, output_path, target_rms=0.01):
    """
    Normalize audio file to target RMS level.

    Args:
        input_path: Path to input WAV file
        output_path: Path to output normalized WAV file
        target_rms: Target RMS level (default: 0.01)

    Returns:
        dict: Statistics about the normalization
    """
    try:
        # Load audio with original sample rate
        audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)

        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio_data ** 2))

        if current_rms == 0:
            print(f"  Warning: {os.path.basename(input_path)} has zero RMS, skipping normalization")
            # Copy file as-is
            audio_normalized = audio_data
            scaling_factor = 0
        else:
            # Calculate scaling factor
            scaling_factor = target_rms / current_rms

            # Apply normalization
            audio_normalized = audio_data * scaling_factor

            # Check for clipping
            peak_amplitude = np.max(np.abs(audio_normalized))
            if peak_amplitude > 1.0:
                print(f"  Warning: {os.path.basename(input_path)} would clip (peak={peak_amplitude:.3f}), applying limiter")
                # Apply limiter to prevent clipping
                audio_normalized = audio_normalized / peak_amplitude * 0.99
                # Recalculate actual RMS after limiting
                actual_rms = np.sqrt(np.mean(audio_normalized ** 2))
            else:
                actual_rms = target_rms

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save normalized audio
        # Handle mono/stereo properly
        if audio_normalized.ndim == 1:
            sf.write(output_path, audio_normalized, sample_rate)
        else:
            # Transpose for soundfile (expects channels last)
            sf.write(output_path, audio_normalized.T, sample_rate)

        return {
            'filename': os.path.basename(input_path),
            'original_rms': current_rms,
            'target_rms': target_rms,
            'actual_rms': actual_rms if current_rms > 0 else 0,
            'scaling_factor': scaling_factor,
            'sample_rate': sample_rate,
            'success': True
        }

    except Exception as e:
        return {
            'filename': os.path.basename(input_path),
            'error': str(e),
            'success': False
        }


def main():
    parser = argparse.ArgumentParser(description='Normalize audio files to target RMS level')
    parser.add_argument('--target-rms', type=float, default=0.01,
                        help='Target RMS level (default: 0.01)')
    parser.add_argument('--input-dir', type=str, default='stim',
                        help='Input directory containing audio files (default: stim)')
    parser.add_argument('--output-dir', type=str, default='stim_normalized',
                        help='Output directory for normalized files (default: stim_normalized)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be normalized without actually processing')

    args = parser.parse_args()

    # Set up paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' not found!")
        return

    # Find all WAV files recursively
    wav_files = list(input_dir.rglob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} WAV files to normalize")
    print(f"Target RMS: {args.target_rms}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    if args.dry_run:
        print("DRY RUN - No files will be modified\n")
        for wav_file in wav_files:
            rel_path = wav_file.relative_to(input_dir)
            output_path = output_dir / rel_path
            print(f"Would normalize: {rel_path}")
            print(f"  -> {output_path}")
        return

    # Process each file
    results = []
    for i, wav_file in enumerate(wav_files, 1):
        # Calculate relative path to preserve directory structure
        rel_path = wav_file.relative_to(input_dir)
        output_path = output_dir / rel_path

        print(f"[{i}/{len(wav_files)}] Processing: {rel_path}")

        result = normalize_audio(str(wav_file), str(output_path), args.target_rms)
        results.append(result)

        if result['success']:
            print(f"  Original RMS: {result['original_rms']:.6f}")
            print(f"  Actual RMS: {result['actual_rms']:.6f}")
            print(f"  Scaling factor: {result['scaling_factor']:.4f}")
        else:
            print(f"  ERROR: {result['error']}")
        print()

    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print("="*80)
    print("NORMALIZATION SUMMARY")
    print("="*80)
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_original_rms = np.mean([r['original_rms'] for r in successful])
        avg_actual_rms = np.mean([r['actual_rms'] for r in successful])
        print(f"\nAverage original RMS: {avg_original_rms:.6f}")
        print(f"Average actual RMS: {avg_actual_rms:.6f}")
        print(f"Target RMS: {args.target_rms:.6f}")

    if failed:
        print("\nFailed files:")
        for r in failed:
            print(f"  {r['filename']}: {r['error']}")

    print(f"\nNormalized files saved to: {output_dir}")


if __name__ == "__main__":
    main()
