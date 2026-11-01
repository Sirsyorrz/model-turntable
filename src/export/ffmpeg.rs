use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    thread::JoinHandle,
};

use anyhow::{Context, Result, bail};

use super::{FrameSink, Progress};
use crate::settings::OutputFormat;

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    }
}

pub fn download_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("model-turntable").join("ffmpeg"))
}

fn command(path: &Path) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn works(path: &Path) -> bool {
    command(path)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

pub fn find_ffmpeg() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(d) = download_dir() {
        candidates.push(d.join(exe_name()));
    }
    if let Some(d) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        candidates.push(d.join(exe_name()));
    }
    candidates.push(PathBuf::from(exe_name()));
    candidates.into_iter().find(|p| works(p))
}

/// Fetches a static ffmpeg build into the app's data directory.
pub fn download_ffmpeg(on_progress: impl Fn(u64, u64)) -> Result<PathBuf> {
    use ffmpeg_sidecar::download::{
        FfmpegDownloadProgressEvent, download_ffmpeg_package_with_progress, ffmpeg_download_url,
        unpack_ffmpeg_without_extras,
    };
    let dir = download_dir().context("no data directory")?;
    std::fs::create_dir_all(&dir)?;
    let url = ffmpeg_download_url()?;
    let archive = download_ffmpeg_package_with_progress(url, &dir, |e| {
        if let FfmpegDownloadProgressEvent::Downloading {
            total_bytes,
            downloaded_bytes,
        } = e
        {
            on_progress(downloaded_bytes, total_bytes);
        }
    })?;
    unpack_ffmpeg_without_extras(&archive, &dir)?;
    let exe = dir.join(exe_name());
    if !works(&exe) {
        bail!("downloaded ffmpeg doesn't run");
    }
    Ok(exe)
}

pub struct FfmpegSink {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr: Option<JoinHandle<String>>,
}

impl FfmpegSink {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        ffmpeg: &Path,
        out: &Path,
        format: OutputFormat,
        width: u32,
        height: u32,
        fps: u32,
        quality: u8,
        alpha: bool,
    ) -> Result<Self> {
        let q = quality.min(100) as f32;
        let size = format!("{width}x{height}");
        let rate = fps.to_string();
        let mut args: Vec<String> = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &size,
            "-framerate",
            &rate,
            "-i",
            "-",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut push = |a: &[&str]| args.extend(a.iter().map(|s| s.to_string()));
        match format {
            OutputFormat::Mp4 => {
                let crf = format!("{}", (40.0 - q * 0.3).round());
                push(&["-c:v", "libx264", "-preset", "slow", "-crf", &crf]);
                push(&["-pix_fmt", "yuv420p", "-movflags", "+faststart"]);
            }
            OutputFormat::Webm => {
                let crf = format!("{}", (50.0 - q * 0.35).round());
                push(&[
                    "-c:v",
                    "libvpx-vp9",
                    "-b:v",
                    "0",
                    "-crf",
                    &crf,
                    "-row-mt",
                    "1",
                ]);
                if alpha {
                    push(&["-pix_fmt", "yuva420p", "-auto-alt-ref", "0"]);
                } else {
                    push(&["-pix_fmt", "yuv420p"]);
                }
            }
            OutputFormat::ProRes => {
                push(&["-c:v", "prores_ks", "-vendor", "apl0"]);
                if alpha {
                    push(&["-profile:v", "4", "-pix_fmt", "yuva444p10le"]);
                } else {
                    push(&["-profile:v", "3", "-pix_fmt", "yuv422p10le"]);
                }
            }
            OutputFormat::DnxHr => {
                push(&[
                    "-c:v",
                    "dnxhd",
                    "-profile:v",
                    "dnxhr_hq",
                    "-pix_fmt",
                    "yuv422p",
                ]);
            }
            OutputFormat::Webp => {
                let quality = format!("{}", q.round());
                push(&[
                    "-c:v",
                    "libwebp_anim",
                    "-lossless",
                    "0",
                    "-quality",
                    &quality,
                    "-loop",
                    "0",
                ]);
                push(&["-pix_fmt", if alpha { "yuva420p" } else { "yuv420p" }]);
            }
            OutputFormat::Apng => {
                push(&[
                    "-c:v", "apng", "-plays", "0", "-pix_fmt", "rgba", "-f", "apng",
                ]);
            }
            OutputFormat::Gif | OutputFormat::PngSequence => unreachable!(),
        }
        args.push(out.to_string_lossy().into_owned());

        let mut child = command(ffmpeg)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting ffmpeg")?;
        let stdin = child.stdin.take();
        let mut stderr = child.stderr.take().context("ffmpeg stderr")?;
        let stderr = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stderr.read_to_string(&mut s);
            s
        });
        Ok(Self {
            child,
            stdin,
            stderr: Some(stderr),
        })
    }

    fn error(&mut self) -> String {
        let _ = self.child.wait();
        self.stderr
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_default()
            .trim()
            .to_owned()
    }
}

impl FrameSink for FfmpegSink {
    fn push(&mut self, rgba: &[u8]) -> Result<()> {
        let stdin = self.stdin.as_mut().context("ffmpeg closed")?;
        if stdin.write_all(rgba).is_err() {
            self.stdin = None;
            bail!("ffmpeg failed: {}", self.error());
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>, _: &Progress) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        let err = self.error();
        if !status.success() {
            bail!("ffmpeg failed: {err}");
        }
        Ok(())
    }
}

impl Drop for FfmpegSink {
    fn drop(&mut self) {
        if self.stdin.is_some() {
            drop(self.stdin.take());
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
