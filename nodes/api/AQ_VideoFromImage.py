import os
import wave
import tempfile
import subprocess
import numpy as np
import torch
from PIL import Image

_KEYSCALES = [
    f"{root} {quality}"
    for quality in ["major", "minor"]
    for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
]


class AQ_StillImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":           ("IMAGE",),
                "audio":           ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "song_cover"}),
                "fps":             ("INT",    {"default": 1, "min": 1, "max": 60, "step": 1}),
                "encoder":         (["auto (gpu→cpu fallback)", "cpu (libx264)", "nvidia (h264_nvenc)", "amd (h264_amf)", "intel (h264_qsv)"],
                                    {"default": "auto (gpu→cpu fallback)"}),
            },
            "optional": {
                "title":    ("STRING", {"default": ""}),
                "tagline":  ("STRING", {"default": ""}),
                "headline": ("STRING", {"default": ""}),
                "tags":     ("STRING", {"default": ""}),
                "keyscale": (_KEYSCALES, {"default": "C major"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "create_video"
    CATEGORY = "Aquasite/Media"

    def create_video(self, image, audio, filename_prefix="song_cover", fps=1,
                     encoder="auto (gpu→cpu fallback)",
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

        # libx264/nvenc require even dimensions
        w, h = pil_image.size
        if w % 2 != 0 or h % 2 != 0:
            pil_image = pil_image.crop((0, 0, w - (w % 2), h - (h % 2)))

        # -- Audio ----------------------------------------------------------------
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        if waveform.ndim == 3:
            waveform = waveform[0]  # [channels, samples]

        # -- Output filename ------------------------------------------------------
        existing = [f for f in os.listdir(output_dir)
                    if f.startswith(filename_prefix) and f.endswith(".mp4")]
        counter = len(existing) + 1
        filename = f"{filename_prefix}_{counter:05d}.mp4"
        output_path = os.path.join(output_dir, filename)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "frame.png")
            wav_path = os.path.join(tmpdir, "audio.wav")

            pil_image.save(img_path, format="PNG")
            self._save_wav(waveform, sample_rate, wav_path)

            meta_args = []
            if title:    meta_args += ["-metadata", f"title={title}"]
            if tagline:  meta_args += ["-metadata", f"comment={tagline}"]
            if tags:     meta_args += ["-metadata", f"genre={tags}"]
            if keyscale: meta_args += ["-metadata", f"key={keyscale}"]
            if headline: meta_args += ["-metadata", f"description={headline}"]

            self._encode(encoder, img_path, wav_path, output_path, fps, meta_args)

        print(f"[AQ_StillImageToVideo] Saved: {output_path}")
        return (output_path,)

    # ------------------------------------------------------------------

    def _encode(self, encoder_choice, img_path, wav_path, output_path, fps, meta_args):
        """Run ffmpeg, trying GPU first when encoder is 'auto'."""

        _GPU_CODECS = {
            "nvidia (h264_nvenc)": self._nvenc_args,
            "amd (h264_amf)":      self._amf_args,
            "intel (h264_qsv)":    self._qsv_args,
        }

        if encoder_choice == "cpu (libx264)":
            cmd = self._build_cmd(img_path, wav_path, output_path, fps,
                                  self._cpu_args(), meta_args)
            self._run(cmd)

        elif encoder_choice in _GPU_CODECS:
            codec_args = _GPU_CODECS[encoder_choice]()
            cmd = self._build_cmd(img_path, wav_path, output_path, fps,
                                  codec_args, meta_args)
            self._run(cmd)

        else:  # auto: try GPU codecs in order, fall back to CPU
            for codec_fn in (self._nvenc_args, self._amf_args, self._qsv_args):
                cmd = self._build_cmd(img_path, wav_path, output_path, fps,
                                      codec_fn(), meta_args)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=6000)
                if result.returncode == 0:
                    return
                print(f"[AQ_StillImageToVideo] GPU encoder failed, trying next: "
                      f"{result.stderr.splitlines()[-1] if result.stderr else ''}")

            # CPU fallback
            print("[AQ_StillImageToVideo] Falling back to CPU (libx264 ultrafast)")
            cmd = self._build_cmd(img_path, wav_path, output_path, fps,
                                  self._cpu_args(), meta_args)
            self._run(cmd)

    def _build_cmd(self, img_path, wav_path, output_path, fps, codec_args, meta_args):
        return [
            "ffmpeg", "-y",
            "-loop", "1",
            "-framerate", str(fps),
            "-i", img_path,
            "-i", wav_path,
        ] + codec_args + [
            "-r", str(fps),
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
        ] + meta_args + [output_path]

    def _run(self, cmd):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=6000)
        if result.returncode != 0:
            print(f"ffmpeg stderr:\n{result.stderr}")
            raise RuntimeError(
                f"ffmpeg failed (code {result.returncode}): {result.stderr[-500:]}"
            )

    # -- Codec argument sets ----------------------------------------------------

    def _cpu_args(self):
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-tune", "stillimage"]

    def _nvenc_args(self):
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "28"]

    def _amf_args(self):
        return ["-c:v", "h264_amf", "-quality", "speed", "-rc", "vbr_latency"]

    def _qsv_args(self):
        return ["-c:v", "h264_qsv", "-preset", "veryfast"]

    # -- WAV helper -------------------------------------------------------------

    def _save_wav(self, waveform, sample_rate, path):
        """Write [channels, samples] float32 numpy array as 16-bit PCM WAV."""
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm = (waveform * 32767).astype(np.int16)

        channels, _ = pcm.shape
        interleaved = pcm.T.flatten()

        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(interleaved.tobytes())
