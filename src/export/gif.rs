use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::Ordering,
};

use anyhow::{Context, Result};
use color_quant::NeuQuant;

use super::{FrameSink, Progress};

const TRANSPARENT: u8 = 255;
const PALETTE_SAMPLES: usize = 1_000_000;

/// Two-pass GIF writer: frames are spooled to disk while a sample of their
/// pixels trains one palette shared by every frame, so shading never flickers
/// between frames the way per-frame palettes do.
pub struct GifSink {
    out: PathBuf,
    spool_path: PathBuf,
    spool: BufWriter<File>,
    width: u32,
    height: u32,
    fps: u32,
    transparent: bool,
    dither: bool,
    matte: [u8; 3],
    frames: u32,
    samples: Vec<u8>,
    sample_stride: usize,
}

impl GifSink {
    pub fn new(
        out: &Path,
        width: u32,
        height: u32,
        fps: u32,
        transparent: bool,
        dither: bool,
        matte: [u8; 3],
    ) -> Result<Self> {
        let spool_path = std::env::temp_dir().join(format!(
            "model-turntable-{}-{}.raw",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let spool = BufWriter::new(File::create(&spool_path).context("creating temp file")?);
        Ok(Self {
            out: out.to_path_buf(),
            spool_path,
            spool,
            width,
            height,
            fps,
            transparent,
            dither,
            matte,
            frames: 0,
            samples: Vec::new(),
            sample_stride: 1,
        })
    }
}

impl FrameSink for GifSink {
    fn push(&mut self, rgba: &[u8]) -> Result<()> {
        let mut frame = rgba.to_vec();
        // One-bit alpha: blend soft edges onto the matte, then threshold.
        for p in frame.as_chunks_mut::<4>().0 {
            let a = p[3] as u32;
            if a == 255 {
                continue;
            }
            for (c, m) in p[..3].iter_mut().zip(self.matte) {
                *c = ((*c as u32 * a + m as u32 * (255 - a) + 127) / 255) as u8;
            }
            p[3] = if !self.transparent || a >= 128 {
                255
            } else {
                0
            };
        }

        if self.frames == 0 {
            let pixels = (self.width * self.height) as usize;
            self.sample_stride = (pixels / (PALETTE_SAMPLES / 60).max(1)).max(1);
        }
        let stride = self.sample_stride;
        let offset = self.frames as usize % stride;
        for p in frame.as_chunks::<4>().0.iter().skip(offset).step_by(stride) {
            if p[3] == 255 {
                self.samples.extend_from_slice(p);
            }
        }
        self.spool.write_all(&frame)?;
        self.frames += 1;
        Ok(())
    }

    fn finish(self: Box<Self>, progress: &Progress) -> Result<()> {
        let GifSink {
            out,
            spool_path,
            spool,
            width,
            height,
            fps,
            transparent,
            dither,
            frames,
            mut samples,
            ..
        } = *self;
        drop(spool.into_inner()?);
        let result = (|| -> Result<()> {
            if samples.is_empty() {
                samples.extend_from_slice(&[0, 0, 0, 255]);
            }
            if samples.len() / 4 > PALETTE_SAMPLES {
                let step = samples.len() / 4 / PALETTE_SAMPLES + 1;
                samples = samples
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .step_by(step)
                    .flatten()
                    .copied()
                    .collect();
            }
            let nq = NeuQuant::new(10, 255, &samples);
            let mut palette = nq.color_map_rgb();
            palette.resize(256 * 3, 0);
            let lut = build_lut(&nq);

            let file = File::create(&out).with_context(|| format!("creating {}", out.display()))?;
            let mut enc =
                ::gif::Encoder::new(BufWriter::new(file), width as u16, height as u16, &palette)?;
            enc.set_repeat(::gif::Repeat::Infinite)?;

            let frame_len = (width * height * 4) as usize;
            let mut reader = BufReader::new(File::open(&spool_path)?);
            let mut rgba = vec![0u8; frame_len];
            let mut prev: Option<Vec<u8>> = None;
            progress.total.store(frames, Ordering::Relaxed);
            for i in 0..frames {
                if progress.cancel.load(Ordering::Relaxed) {
                    anyhow::bail!("Cancelled");
                }
                reader.read_exact(&mut rgba)?;
                let indices = quantize(&rgba, width, &lut, dither);
                let delay = frame_delay(i, fps);

                let frame = if transparent {
                    full_frame(
                        indices.clone(),
                        width,
                        height,
                        delay,
                        ::gif::DisposalMethod::Background,
                    )
                } else {
                    match &prev {
                        Some(p) => delta_frame(p, &indices, width, height, delay),
                        None => full_frame(
                            indices.clone(),
                            width,
                            height,
                            delay,
                            ::gif::DisposalMethod::Keep,
                        ),
                    }
                };
                enc.write_frame(&frame)?;
                prev = Some(indices);
                progress.done.store(i + 1, Ordering::Relaxed);
            }
            Ok(())
        })();
        let _ = std::fs::remove_file(&spool_path);
        result
    }
}

/// Delays are in centiseconds, so spread rounding error across frames to keep
/// the overall speed exact. Browsers treat delays under 2cs as 10cs.
fn frame_delay(i: u32, fps: u32) -> u16 {
    let t = |n: u32| (n as f64 * 100.0 / fps as f64).round() as i64;
    (t(i + 1) - t(i)).max(2) as u16
}

fn build_lut(nq: &NeuQuant) -> Vec<u8> {
    let mut lut = vec![0u8; 64 * 64 * 64];
    for r in 0..64u32 {
        for g in 0..64u32 {
            for b in 0..64u32 {
                let px = [
                    (r << 2 | r >> 4) as u8,
                    (g << 2 | g >> 4) as u8,
                    (b << 2 | b >> 4) as u8,
                    255,
                ];
                lut[((r << 12) | (g << 6) | b) as usize] = nq.index_of(&px) as u8;
            }
        }
    }
    lut
}

const BAYER: [[i32; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

fn quantize(rgba: &[u8], width: u32, lut: &[u8], dither: bool) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p[3] == 0 {
                return TRANSPARENT;
            }
            let (x, y) = (i as u32 % width, i as u32 / width);
            let d = if dither {
                (BAYER[(y & 3) as usize][(x & 3) as usize] - 8) * 3 / 4
            } else {
                0
            };
            let q = |c: u8| ((c as i32 + d).clamp(0, 255) as u32) >> 2;
            lut[((q(p[0]) << 12) | (q(p[1]) << 6) | q(p[2])) as usize]
        })
        .collect()
}

fn full_frame(
    indices: Vec<u8>,
    width: u32,
    height: u32,
    delay: u16,
    dispose: ::gif::DisposalMethod,
) -> ::gif::Frame<'static> {
    ::gif::Frame {
        width: width as u16,
        height: height as u16,
        buffer: indices.into(),
        delay,
        dispose,
        transparent: Some(TRANSPARENT),
        ..Default::default()
    }
}

/// Only stores the changed rectangle, with unchanged pixels inside it made
/// transparent so they compress to nearly nothing.
fn delta_frame(
    prev: &[u8],
    cur: &[u8],
    width: u32,
    height: u32,
    delay: u16,
) -> ::gif::Frame<'static> {
    let (w, h) = (width as usize, height as usize);
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if prev[i] != cur[i] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 > x1 {
        (x0, y0, x1, y1) = (0, 0, 0, 0);
    }
    let (fw, fh) = (x1 - x0 + 1, y1 - y0 + 1);
    let mut buf = Vec::with_capacity(fw * fh);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            buf.push(if prev[i] == cur[i] {
                TRANSPARENT
            } else {
                cur[i]
            });
        }
    }
    ::gif::Frame {
        left: x0 as u16,
        top: y0 as u16,
        width: fw as u16,
        height: fh as u16,
        buffer: buf.into(),
        delay,
        dispose: ::gif::DisposalMethod::Keep,
        transparent: Some(TRANSPARENT),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_average_to_fps() {
        let total: u32 = (0..30).map(|i| frame_delay(i, 30) as u32).sum();
        assert_eq!(total, 100);
    }
}
