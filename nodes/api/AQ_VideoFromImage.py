import os
import wave
import tempfile
import subprocess
import numpy as np
import torch
from PIL import Image


class AQ_StillImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":           ("IMAGE",),
                "audio":           ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "song_cover"}),
            },
            "optional": {
                "title":    ("STRING", {"default": ""}),
                "tagline":  ("STRING", {"default": ""}),
                "headline": ("STRING", {"default": ""}),
                "tags":     ("STRING", {"default": ""}),
                "keyscale": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "create_video"
    CATEGORY = "Aquasite/Media"

    def create_video(self, image, audio, filename_prefix="song_cover",
                     title="", tagline="", headline="", tags="", keyscale=""):
        import folder_paths

        output_dir = folder_paths.get_output_directory()

        # -- Image ----------------------------------------------------------------
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
        else:
            img_np = np.array(image)

        if img_np.ndim == 4:
            img_np = img_np[0]

        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)

        pil_image = Image.fromarray(img_np)

        # ffmpeg libx264 requires even dimensions
        w, h = pil_image.size
        if w % 2 != 0 or h % 2 != 0:
            pil_image = pil_image.crop((0, 0, w - (w % 2), h - (h % 2)))

        # -- Audio ----------------------------------------------------------------
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        if waveform.ndim == 3:
            waveform = waveform[0]          # [channels, samples]

        # -- Output filename ------------------------------------------------------
        existing = [f for f in os.listdir(output_dir) if f.startswith(filename_prefix) and f.endswith(".mp4")]
        counter = len(existing) + 1
        filename = f"{filename_prefix}_{counter:05d}.mp4"
        output_path = os.path.join(output_dir, filename)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "frame.png")
            wav_path = os.path.join(tmpdir, "audio.wav")

            pil_image.save(img_path, format="PNG")
            self._save_wav(waveform, sample_rate, wav_path)

            # ffmpeg metadata
            meta_args = []
            if title:
                meta_args += ["-metadata", f"title={title}"]
            if tagline:
                meta_args += ["-metadata", f"comment={tagline}"]
            if tags:
                meta_args += ["-metadata", f"genre={tags}"]
            if keyscale:
                meta_args += ["-metadata", f"key={keyscale}"]
            if headline:
                meta_args += ["-metadata", f"description={headline}"]

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", img_path,
                "-i", wav_path,
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
            ] + meta_args + [output_path]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"ffmpeg stderr:\n{result.stderr}")
                raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {result.stderr[-500:]}")

        print(f"[AQ_StillImageToVideo] Saved: {output_path}")
        return (output_path,)

    def _save_wav(self, waveform, sample_rate, path):
        """Write [channels, samples] float32 numpy array as 16-bit PCM WAV."""
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm = (waveform * 32767).astype(np.int16)

        channels, n_samples = pcm.shape
        # Interleave: shape (n_samples, channels) → flatten
        interleaved = pcm.T.flatten()

        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)          # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(interleaved.tobytes())
