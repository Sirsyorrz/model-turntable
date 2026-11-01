mod ffmpeg;
mod gif;

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

use anyhow::{Result, bail};

pub use ffmpeg::{download_ffmpeg, find_ffmpeg};

use crate::{
    render::{FrameResources, GpuModel, Renderer, scene},
    settings::{OutputFormat, Settings},
};

#[derive(Default)]
pub struct Progress {
    pub done: AtomicU32,
    pub total: AtomicU32,
    pub cancel: AtomicBool,
    pub stage: Mutex<String>,
}

impl Progress {
    fn set_stage(&self, s: &str) {
        *self.stage.lock().unwrap() = s.to_owned();
    }

    pub fn stage(&self) -> String {
        self.stage.lock().unwrap().clone()
    }

    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed).max(1);
        self.done.load(Ordering::Relaxed) as f32 / t as f32
    }
}

pub struct Job {
    pub renderer: Arc<Renderer>,
    pub model: Arc<GpuModel>,
    pub resources: FrameResources,
    pub framing: scene::Framing,
    pub flat: bool,
    pub settings: Settings,
    pub out_path: PathBuf,
    pub ffmpeg: Option<PathBuf>,
}

trait FrameSink {
    /// `rgba` is straight (non-premultiplied) sRGB RGBA.
    fn push(&mut self, rgba: &[u8]) -> Result<()>;
    fn finish(self: Box<Self>, progress: &Progress) -> Result<()>;
}

pub fn run(job: Job, progress: &Progress) -> Result<PathBuf> {
    let s = &job.settings;
    let format = s.output.format;
    let (w, h) = s.output.render_size();
    let frames = s.output.frame_count(s.motion.duration);
    let transparent = s.background.transparent && format.supports_alpha();
    let fps = s.output.fps.max(1);

    let mut sink: Box<dyn FrameSink> = match format {
        OutputFormat::Gif => Box::new(gif::GifSink::new(
            &job.out_path,
            w,
            h,
            fps,
            transparent,
            s.output.gif_dither,
            s.background.color,
        )?),
        OutputFormat::PngSequence => Box::new(PngSequence::new(&job.out_path, w, h)?),
        _ => {
            let Some(ffmpeg) = &job.ffmpeg else {
                bail!("{} export needs ffmpeg, which wasn't found", format.label());
            };
            Box::new(ffmpeg::FfmpegSink::spawn(
                ffmpeg,
                &job.out_path,
                format,
                w,
                h,
                fps,
                s.output.quality,
                transparent,
            )?)
        }
    };

    progress.total.store(frames, Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    progress.set_stage("Rendering");

    let r = &job.renderer;
    let target = r.create_target(w, h, s.output.supersample);
    let draw_ground = s.background.ground_shadow;
    for i in 0..frames {
        if progress.cancel.load(Ordering::Relaxed) {
            bail!("Cancelled");
        }
        let t = i as f32 / frames as f32;
        let uniforms = scene::build(scene::SceneInputs {
            settings: s,
            framing: &job.framing,
            env: &job.resources.env,
            flat: job.flat,
            aspect: w as f32 / h as f32,
            angle_deg: s.motion.angle_at(t),
            transparent,
        });
        r.render(
            &target,
            &job.model,
            &job.resources,
            &uniforms,
            !transparent,
            draw_ground,
        );
        let mut pixels = r.read_target(&target);
        unpremultiply(&mut pixels);
        sink.push(&pixels)?;
        progress.done.store(i + 1, Ordering::Relaxed);
    }
    progress.set_stage("Encoding");
    sink.finish(progress)?;
    Ok(job.out_path)
}

/// The renderer writes linear premultiplied colour into an sRGB target, so
/// un-premultiply in linear space.
fn unpremultiply(rgba: &mut [u8]) {
    let to_lin: Vec<f32> = (0..256)
        .map(|i| crate::settings::srgb_to_linear(i as u8))
        .collect();
    for p in rgba.as_chunks_mut::<4>().0 {
        let a = p[3];
        if a == 255 || a == 0 {
            continue;
        }
        let af = a as f32 / 255.0;
        for c in &mut p[..3] {
            *c = linear_to_srgb((to_lin[*c as usize] / af).min(1.0));
        }
    }
}

pub fn linear_to_srgb(v: f32) -> u8 {
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

struct PngSequence {
    dir: PathBuf,
    index: u32,
    width: u32,
    height: u32,
}

impl PngSequence {
    fn new(dir: &Path, width: u32, height: u32) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            index: 0,
            width,
            height,
        })
    }
}

impl FrameSink for PngSequence {
    fn push(&mut self, rgba: &[u8]) -> Result<()> {
        self.index += 1;
        let path = self.dir.join(format!("frame_{:05}.png", self.index));
        let Some(img) = image::RgbaImage::from_raw(self.width, self.height, rgba.to_vec()) else {
            bail!("frame size mismatch");
        };
        img.save(path)?;
        Ok(())
    }

    fn finish(self: Box<Self>, _: &Progress) -> Result<()> {
        Ok(())
    }
}
