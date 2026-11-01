//! Self-updater compatible with the Tauri updater used by soundbox and chaptr:
//! the same `latest.json` on GitHub releases, signed with a Tauri/minisign key.

use std::{
    io::Read,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use serde::Deserialize;

const ENDPOINT: &str =
    "https://github.com/Sirsyorrz/model-turntable/releases/latest/download/latest.json";
const PUBKEY: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IDE3MkIxRjY4RDAyQTFBRUUKUldUdUdpclFhQjhyRjdZMEEvVmJTckEzNnYvTE4xWHRxUUtObThMaHhmYXpva25URkVJK3Y3UmkK";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Only CI release builds update themselves, so a local `cargo build` never
/// gets overwritten by a download.
const ENABLED: bool = option_env!("MT_UPDATER").is_some();

#[cfg(target_os = "linux")]
const PLATFORM: &str = "linux-x86_64";
#[cfg(windows)]
const PLATFORM: &str = "windows-x86_64";
#[cfg(not(any(target_os = "linux", windows)))]
const PLATFORM: &str = "unsupported";

#[derive(Clone)]
pub enum Phase {
    Idle,
    Checking,
    Found(Release),
    Downloading { done: u64, total: u64 },
    Ready,
    Error(String),
}

#[derive(Clone)]
pub struct Release {
    pub version: String,
    url: String,
    signature: String,
}

#[derive(Deserialize)]
struct Manifest {
    version: String,
    platforms: std::collections::HashMap<String, PlatformEntry>,
}

#[derive(Deserialize)]
struct PlatformEntry {
    url: String,
    signature: String,
}

#[derive(Clone)]
pub struct Updater {
    phase: Arc<Mutex<Phase>>,
    restart: Arc<AtomicBool>,
    repaint: egui::Context,
}

impl Updater {
    pub fn new(ctx: egui::Context, restart: Arc<AtomicBool>) -> Updater {
        cleanup_previous();
        Updater {
            phase: Arc::new(Mutex::new(Phase::Idle)),
            restart,
            repaint: ctx,
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase.lock().unwrap().clone()
    }

    fn set(&self, p: Phase) {
        *self.phase.lock().unwrap() = p;
        self.repaint.request_repaint();
    }

    /// `loud` reports "already up to date" and failures. The check on startup
    /// stays silent so a flaky network never nags.
    pub fn look(&self, loud: bool) {
        if matches!(self.phase(), Phase::Checking | Phase::Downloading { .. }) {
            return;
        }
        if !ENABLED {
            if loud {
                self.set(Phase::Error("updates are off in local builds".into()));
            }
            return;
        }
        self.set(Phase::Checking);
        let this = self.clone();
        std::thread::spawn(move || {
            let next = match check() {
                Ok(Some(r)) => Phase::Found(r),
                Ok(None) if loud => Phase::Error("already up to date".into()),
                Err(e) if loud => Phase::Error(format!("update check failed: {e:#}")),
                _ => Phase::Idle,
            };
            this.set(next);
        });
    }

    pub fn install(&self) {
        let Phase::Found(release) = self.phase() else {
            return;
        };
        self.set(Phase::Downloading { done: 0, total: 0 });
        let this = self.clone();
        std::thread::spawn(move || {
            let next = match download_and_install(&release, |done, total| {
                *this.phase.lock().unwrap() = Phase::Downloading { done, total };
                this.repaint.request_repaint();
            }) {
                Ok(()) => Phase::Ready,
                Err(e) => Phase::Error(format!("update failed: {e:#}")),
            };
            this.set(next);
        });
    }

    pub fn dismiss(&self) {
        if !matches!(self.phase(), Phase::Downloading { .. }) {
            self.set(Phase::Idle);
        }
    }

    /// Closes the window; [`relaunch_if_requested`] starts the new binary once
    /// eframe has saved state and exited.
    pub fn restart(&self) {
        self.restart.store(true, Ordering::Relaxed);
        self.repaint.send_viewport_cmd(egui::ViewportCommand::Close);
    }
}

pub fn relaunch_if_requested(flag: &AtomicBool) {
    if !flag.load(Ordering::Relaxed) {
        return;
    }
    if let Ok(target) = install_target() {
        let _ = std::process::Command::new(target).spawn();
    }
}

fn check() -> Result<Option<Release>> {
    // Overridable so the whole flow can be tested against a local server.
    let endpoint = std::env::var("MT_UPDATE_ENDPOINT").unwrap_or_else(|_| ENDPOINT.to_owned());
    let manifest: Manifest =
        serde_json::from_str(&ureq::get(&endpoint).call()?.body_mut().read_to_string()?)
            .context("reading latest.json")?;
    let latest = semver::Version::parse(manifest.version.trim_start_matches('v'))?;
    if latest <= semver::Version::parse(VERSION)? {
        return Ok(None);
    }
    let entry = manifest
        .platforms
        .get(PLATFORM)
        .ok_or_else(|| anyhow!("no {PLATFORM} build in this release"))?;
    install_target()?;
    Ok(Some(Release {
        version: latest.to_string(),
        url: entry.url.clone(),
        signature: entry.signature.clone(),
    }))
}

/// The file that gets replaced: the AppImage on Linux, the exe on Windows.
fn install_target() -> Result<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        std::env::var_os("APPIMAGE")
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("only the AppImage can update itself"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(std::env::current_exe()?)
    }
}

fn download_and_install(release: &Release, progress: impl Fn(u64, u64)) -> Result<()> {
    let target = install_target()?;
    let mut resp = ureq::get(&release.url).call()?;
    let total = resp.body().content_length().unwrap_or(0);
    let mut reader = resp.body_mut().as_reader();
    let mut data = Vec::with_capacity(total as usize);
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        progress(data.len() as u64, total);
    }
    verify(&data, &release.signature)?;
    replace(&target, &data)
}

/// Tauri stores both the key and the signature as base64 of the minisign text.
fn verify(data: &[u8], signature: &str) -> Result<()> {
    let b64 = base64::engine::general_purpose::STANDARD;
    let key = String::from_utf8(b64.decode(PUBKEY)?)?;
    let sig = String::from_utf8(b64.decode(signature.trim())?)?;
    let key = minisign_verify::PublicKey::decode(&key).map_err(|e| anyhow!("{e}"))?;
    let sig = minisign_verify::Signature::decode(&sig).map_err(|e| anyhow!("{e}"))?;
    key.verify(data, &sig, true)
        .map_err(|_| anyhow!("signature doesn't match, refusing to install"))
}

fn replace(target: &std::path::Path, data: &[u8]) -> Result<()> {
    let staged = target.with_extension("update");
    std::fs::write(&staged, data)
        .with_context(|| format!("writing next to {}", target.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))?;
    }
    // Windows can't overwrite a running exe but can rename it out of the way.
    #[cfg(windows)]
    {
        let old = target.with_extension("old");
        let _ = std::fs::remove_file(&old);
        std::fs::rename(target, &old)?;
    }
    if let Err(e) = std::fs::rename(&staged, target) {
        let _ = std::fs::remove_file(&staged);
        bail!("replacing {}: {e}", target.display());
    }
    Ok(())
}

fn cleanup_previous() {
    #[cfg(windows)]
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::fs::remove_file(exe.with_extension("old"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Signed with `tauri signer sign` and the release key.
    const SIGNED: &[u8] = b"model-turntable update test\n";
    const SIGNATURE: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpIHNlY3JldCBrZXkKUlVUdUdpclFhQjhyRjdDWlZ6YklIU0hwOXpjZWVLUmQzQTFvQUl0aHcrU3NaUWs4VldpQ3Y4NVVPSy84VVlPTlBTTEdmcVVYWHpCUitzTTdoWlBBVjNWc05aWUZ6V2t6bVEwPQp0cnVzdGVkIGNvbW1lbnQ6IHRpbWVzdGFtcDoxNzkzNTE2NDI5CWZpbGU6bXQtc2lnbi50eHQKczhDekR6ODdJTE1GT2NaWXBmL1BaY25yZXFDdUliai9zS0VBMWtnUnE3Zmt4SzVrcVg0OWVyWlFQRjZ0MHVBREZOVzU3SlVpc3pCcU1VUGF0UGdJQ3c9PQo=";

    #[test]
    fn accepts_tauri_signature() {
        verify(SIGNED, SIGNATURE).unwrap();
    }

    #[test]
    #[ignore]
    fn e2e_local() {
        let r = check().unwrap().expect("update found");
        download_and_install(&r, |_, _| {}).unwrap();
    }

    #[test]
    fn rejects_tampered_data() {
        assert!(verify(b"model-turntable update test!\n", SIGNATURE).is_err());
    }
}
