use std::{
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

use crate::{
    export, lighting,
    model::Model,
    render::{Gpu, Renderer},
    session::Session,
    settings::{
        BackgroundStyle, LightingMode, MotionKind, OutputFormat, Settings, UpAxis, parse_hex_color,
    },
};

#[derive(Parser)]
#[command(version, about = "Turntable GIFs and videos of 3D models")]
pub struct Args {
    /// Model to open in the GUI
    pub model: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Render a turntable without opening the window
    Render(Box<RenderArgs>),
    /// List built-in HDRIs, light presets and matcaps
    List,
}

#[derive(Clone, Copy, ValueEnum)]
enum Lighting {
    Studio,
    Lights,
    Matcap,
}

#[derive(Clone, Copy, ValueEnum)]
enum BgStyle {
    Solid,
    Gradient,
    Radial,
    Hdri,
}

#[derive(Clone, Copy, ValueEnum)]
enum Up {
    Auto,
    Y,
    Z,
}

#[derive(clap::Args)]
pub struct RenderArgs {
    model: PathBuf,
    /// Output file (format from extension) or folder for PNG sequences
    #[arg(short, long)]
    out: PathBuf,
    /// Start from settings saved in the GUI (JSON); flags override them
    #[arg(long)]
    settings: Option<PathBuf>,
    /// WIDTHxHEIGHT, e.g. 600x600 or 1920x1080
    #[arg(long)]
    size: Option<String>,
    #[arg(long)]
    fps: Option<u32>,
    /// Seconds per loop
    #[arg(long)]
    duration: Option<f32>,
    /// Write a PNG sequence into --out
    #[arg(long)]
    png_sequence: bool,
    /// Write DNxHR instead of ProRes for .mov
    #[arg(long)]
    dnxhr: bool,
    #[arg(long, value_enum)]
    lighting: Option<Lighting>,
    /// HDRI id for studio lighting (see `list`)
    #[arg(long)]
    hdri: Option<String>,
    /// Light preset name (see `list`)
    #[arg(long)]
    preset: Option<String>,
    #[arg(long)]
    matcap: Option<String>,
    /// Spin the lighting with the model
    #[arg(long)]
    follow: bool,
    /// Exposure in stops
    #[arg(long, allow_hyphen_values = true)]
    exposure: Option<f32>,
    /// Camera elevation in degrees
    #[arg(long, allow_hyphen_values = true)]
    elevation: Option<f32>,
    #[arg(long)]
    zoom: Option<f32>,
    #[arg(long)]
    fov: Option<f32>,
    /// Background colour, #rrggbb
    #[arg(long)]
    bg: Option<String>,
    /// Second gradient colour, #rrggbb
    #[arg(long)]
    bg2: Option<String>,
    #[arg(long, value_enum)]
    bg_style: Option<BgStyle>,
    #[arg(long)]
    transparent: bool,
    #[arg(long)]
    no_shadow: bool,
    /// Ignore materials and render in clay
    #[arg(long)]
    clay: bool,
    #[arg(long, value_enum)]
    up: Option<Up>,
    /// Model rotation in degrees, X,Y,Z
    #[arg(long, allow_hyphen_values = true)]
    rotate: Option<String>,
    /// Swing back and forth by this many degrees instead of spinning
    #[arg(long)]
    swing: Option<f32>,
    #[arg(long)]
    clockwise: bool,
    /// 1-4, renders larger and scales down for smoother edges
    #[arg(long)]
    supersample: Option<u32>,
    /// 0-100, for MP4/WebM/WebP
    #[arg(long)]
    quality: Option<u8>,
    /// Ordered dithering for GIFs
    #[arg(long)]
    dither: bool,
}

pub fn run(cmd: Command) -> Result<()> {
    match cmd {
        Command::List => {
            println!("HDRIs:");
            for h in lighting::HDRIS {
                println!("  {:<14} {}", h.id, h.label);
            }
            println!("Light presets:");
            for p in lighting::PRESETS {
                println!("  {}", p.name);
            }
            println!("Matcaps:");
            for m in lighting::MATCAPS {
                println!("  {m}");
            }
            Ok(())
        }
        Command::Render(a) => render(*a),
    }
}

fn render(a: RenderArgs) -> Result<()> {
    let mut s: Settings = match &a.settings {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)
            .with_context(|| format!("reading {}", p.display()))?,
        None => Settings::default(),
    };
    apply(&a, &mut s)?;

    let t0 = Instant::now();
    let model = Model::load(&a.model)?;
    for w in &model.warnings {
        eprintln!("warning: {w}");
    }
    eprintln!(
        "loaded {} ({} triangles) in {:.2}s",
        model.file_name(),
        model.triangle_count(),
        t0.elapsed().as_secs_f32()
    );

    let gpu = Gpu::headless()?;
    eprintln!("gpu: {}", gpu.adapter_name);
    let mut session = Session::new(Arc::new(Renderer::new(gpu)));
    session.set_model(model);

    let ffmpeg = if s.output.format.needs_ffmpeg() {
        Some(export::find_ffmpeg().context("ffmpeg not found on PATH")?)
    } else {
        None
    };
    let job = session
        .job(&s, a.out.clone(), ffmpeg)
        .context("preparing render")?;
    let progress = Arc::new(export::Progress::default());
    let p = progress.clone();
    let worker = std::thread::spawn(move || export::run(job, &p));
    let mut last = String::new();
    while !worker.is_finished() {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let line = format!(
            "{} {}/{}",
            progress.stage(),
            progress.done.load(Ordering::Relaxed),
            progress.total.load(Ordering::Relaxed)
        );
        if line != last {
            eprint!("\r{line}    ");
            last = line;
        }
    }
    eprintln!();
    let out = worker
        .join()
        .map_err(|_| anyhow::anyhow!("render thread panicked"))??;
    eprintln!(
        "wrote {} in {:.1}s",
        out.display(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

fn apply(a: &RenderArgs, s: &mut Settings) -> Result<()> {
    if let Some(size) = &a.size {
        let (w, h) = size
            .split_once(['x', 'X'])
            .and_then(|(w, h)| Some((w.trim().parse().ok()?, h.trim().parse().ok()?)))
            .context("--size must look like 600x600")?;
        s.output.width = w;
        s.output.height = h;
    }
    if a.png_sequence {
        s.output.format = OutputFormat::PngSequence;
    } else if a.dnxhr {
        s.output.format = OutputFormat::DnxHr;
    } else {
        let ext = a.out.extension().and_then(|e| e.to_str()).unwrap_or("");
        s.output.format = OutputFormat::from_extension(ext).with_context(|| {
            format!("can't tell the output format from .{ext}, use gif/mp4/webm/mov/webp/png")
        })?;
    }
    if let Some(v) = a.fps {
        s.output.fps = v.clamp(1, 120);
    }
    if let Some(v) = a.duration {
        s.motion.duration = v.max(0.1);
    }
    if let Some(v) = a.supersample {
        s.output.supersample = v.clamp(1, 4);
    }
    if let Some(v) = a.quality {
        s.output.quality = v.min(100);
    }
    s.output.gif_dither |= a.dither;
    if let Some(l) = a.lighting {
        s.lighting.mode = match l {
            Lighting::Studio => LightingMode::Studio,
            Lighting::Lights => LightingMode::Lights,
            Lighting::Matcap => LightingMode::Matcap,
        };
    }
    if let Some(h) = &a.hdri {
        if !lighting::HDRIS.iter().any(|x| x.id == h) {
            bail!("unknown HDRI {h}, see `list`");
        }
        s.lighting.hdri = h.clone();
    }
    if let Some(p) = &a.preset {
        let preset = lighting::PRESETS
            .iter()
            .find(|x| x.name.to_lowercase().starts_with(&p.to_lowercase()))
            .with_context(|| format!("unknown preset {p}, see `list`"))?;
        s.lighting.preset = preset.name.to_owned();
        s.lighting.follow_model = preset.follow_model;
        if a.lighting.is_none() {
            s.lighting.mode = LightingMode::Lights;
        }
    }
    if let Some(m) = &a.matcap {
        let name = lighting::MATCAPS
            .iter()
            .find(|x| x.eq_ignore_ascii_case(m))
            .with_context(|| format!("unknown matcap {m}, see `list`"))?;
        s.lighting.matcap = (*name).to_owned();
        if a.lighting.is_none() {
            s.lighting.mode = LightingMode::Matcap;
        }
    }
    s.lighting.follow_model |= a.follow;
    if let Some(v) = a.exposure {
        s.lighting.exposure = v;
    }
    if let Some(v) = a.elevation {
        s.camera.elevation = v;
    }
    if let Some(v) = a.zoom {
        s.camera.zoom = v;
    }
    if let Some(v) = a.fov {
        s.camera.fov = v;
    }
    if let Some(c) = &a.bg {
        s.background.color = parse_hex_color(c).context("--bg must be #rrggbb")?;
        if a.bg_style.is_none() && a.bg2.is_none() {
            s.background.style = BackgroundStyle::Solid;
        }
    }
    if let Some(c) = &a.bg2 {
        s.background.color2 = parse_hex_color(c).context("--bg2 must be #rrggbb")?;
    }
    if let Some(b) = a.bg_style {
        s.background.style = match b {
            BgStyle::Solid => BackgroundStyle::Solid,
            BgStyle::Gradient => BackgroundStyle::Vertical,
            BgStyle::Radial => BackgroundStyle::Radial,
            BgStyle::Hdri => BackgroundStyle::Hdri,
        };
    }
    s.background.transparent |= a.transparent;
    if a.no_shadow {
        s.background.ground_shadow = false;
    }
    s.model.clay |= a.clay;
    if let Some(u) = a.up {
        s.model.up = match u {
            Up::Auto => UpAxis::Auto,
            Up::Y => UpAxis::Y,
            Up::Z => UpAxis::Z,
        };
    }
    if let Some(r) = &a.rotate {
        let v: Vec<f32> = r
            .split(',')
            .map(|x| x.trim().parse())
            .collect::<std::result::Result<_, _>>()
            .context("--rotate must look like 0,90,0")?;
        if v.len() != 3 {
            bail!("--rotate needs three values");
        }
        s.model.rotation = [v[0], v[1], v[2]];
    }
    if let Some(v) = a.swing {
        s.motion.kind = MotionKind::Swing;
        s.motion.swing_angle = v;
    }
    s.motion.clockwise |= a.clockwise;
    Ok(())
}
