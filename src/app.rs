use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, channel},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use anyhow::anyhow;
use egui::{
    Align, Align2, Button, Color32, CornerRadius, FontId, Frame, Layout, Margin, Rect, RichText,
    Sense, Stroke, StrokeKind, TextureHandle, Ui, UiBuilder, pos2, vec2,
};
use serde::{Deserialize, Serialize};

use crate::{
    export,
    lighting::{HDRIS, MATCAPS, PRESETS},
    model::{EXTENSIONS, Model},
    render::{Gpu, RenderTarget, Renderer, env::HdrImage},
    session::Session,
    settings::*,
    theme::{self, *},
};

const LABEL_W: f32 = 92.0;
const MAX_RECENT: usize = 8;

pub fn run(initial: Option<PathBuf>) -> anyhow::Result<()> {
    let icon = image::load_from_memory(include_bytes!("../assets/icon.png"))
        .map(|i| {
            let i = i.into_rgba8();
            egui::IconData {
                width: i.width(),
                height: i.height(),
                rgba: i.into_raw(),
            }
        })
        .ok();
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("model-turntable")
        .with_app_id("model-turntable")
        .with_inner_size([1320.0, 840.0])
        .with_min_inner_size([920.0, 580.0]);
    if let Some(icon) = icon {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        event_loop_builder: x11_for_drag_and_drop(),
        ..Default::default()
    };
    eframe::run_native(
        "model-turntable",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, initial)?))),
    )
    .map_err(|e| anyhow!("{e}"))
}

/// winit has no drag-and-drop on Wayland, so prefer XWayland when it's
/// available. `MT_WAYLAND=1` opts back into native Wayland.
#[cfg(target_os = "linux")]
fn x11_for_drag_and_drop() -> Option<eframe::EventLoopBuilderHook> {
    if std::env::var_os("DISPLAY").is_none() || std::env::var_os("MT_WAYLAND").is_some() {
        return None;
    }
    Some(Box::new(|builder| {
        use winit::platform::x11::EventLoopBuilderExtX11;
        builder.with_x11();
    }))
}

#[cfg(not(target_os = "linux"))]
fn x11_for_drag_and_drop() -> Option<eframe::EventLoopBuilderHook> {
    None
}

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
struct Persisted {
    settings: Settings,
    recent: Vec<PathBuf>,
    export_dir: Option<PathBuf>,
    closed_sections: BTreeSet<String>,
}

struct ModelInfo {
    path: PathBuf,
    name: String,
    triangles: usize,
    vertices: usize,
    materials: usize,
    textures: usize,
    warnings: Vec<String>,
    load_time: Duration,
}

struct Loading {
    path: PathBuf,
    started: Instant,
    rx: Receiver<anyhow::Result<Model>>,
}

struct ExportState {
    progress: Arc<export::Progress>,
    handle: Option<JoinHandle<anyhow::Result<PathBuf>>>,
    started: Instant,
}

struct Preview {
    target: RenderTarget,
    texture: egui::TextureId,
    rendered: Option<(Settings, f32, bool)>,
}

enum FfmpegState {
    Found(PathBuf),
    Missing,
    Downloading {
        progress: Arc<(AtomicU64, AtomicU64)>,
        rx: Receiver<anyhow::Result<PathBuf>>,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum Tone {
    Info,
    Ok,
    Bad,
}

struct Status {
    text: String,
    tone: Tone,
    reveal: Option<PathBuf>,
}

pub struct App {
    session: Session,
    render_state: egui_wgpu::RenderState,
    p: Persisted,
    info: Option<ModelInfo>,
    loading: Option<Loading>,
    preview: Option<Preview>,
    playing: bool,
    t: f32,
    export: Option<ExportState>,
    ffmpeg: FfmpegState,
    status: Option<Status>,
    hdri_thumbs: Vec<TextureHandle>,
    matcap_thumbs: Vec<TextureHandle>,
    screenshot: Option<Screenshot>,
}

/// `MT_SCREENSHOT=out.png` saves the window once the first model has rendered
/// and quits. Used for README images.
struct Screenshot {
    path: PathBuf,
    frames: u32,
    done: Arc<std::sync::atomic::AtomicBool>,
}

impl App {
    fn new(
        cc: &eframe::CreationContext,
        initial: Option<PathBuf>,
    ) -> Result<App, Box<dyn std::error::Error + Send + Sync>> {
        theme::apply(&cc.egui_ctx);
        let rs = cc
            .wgpu_render_state
            .clone()
            .ok_or("wgpu renderer unavailable")?;
        let gpu = Gpu {
            device: rs.device.clone(),
            queue: rs.queue.clone(),
            adapter_name: rs.adapter.get_info().name,
        };
        let mut session = Session::new(Arc::new(Renderer::new(gpu)));
        let mut p: Persisted = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();
        p.recent.retain(|r| r.is_absolute());

        let hdri_thumbs = HDRIS
            .iter()
            .map(|h| {
                let img = HdrImage::decode(h.data).expect("bundled HDRI decodes");
                cc.egui_ctx.load_texture(
                    format!("hdri-{}", h.id),
                    hdri_thumbnail(&img, 160, 80),
                    egui::TextureOptions::LINEAR,
                )
            })
            .collect();
        let matcap_thumbs = MATCAPS
            .iter()
            .map(|m| {
                let px = session.matcap_pixels(m);
                cc.egui_ctx.load_texture(
                    format!("matcap-{m}"),
                    matcap_thumbnail(&px),
                    egui::TextureOptions::LINEAR,
                )
            })
            .collect();

        let ffmpeg = match export::find_ffmpeg() {
            Some(p) => FfmpegState::Found(p),
            None => FfmpegState::Missing,
        };

        let mut app = App {
            session,
            render_state: rs,
            p,
            info: None,
            loading: None,
            preview: None,
            playing: true,
            t: 0.0,
            export: None,
            ffmpeg,
            status: None,
            hdri_thumbs,
            matcap_thumbs,
            screenshot: std::env::var_os("MT_SCREENSHOT").map(|p| Screenshot {
                path: p.into(),
                frames: 0,
                done: Default::default(),
            }),
        };
        if let Some(path) = initial {
            app.open(path);
        }
        Ok(app)
    }

    fn settings(&self) -> &Settings {
        &self.p.settings
    }

    fn set_status(&mut self, text: impl Into<String>, tone: Tone) {
        self.status = Some(Status {
            text: text.into(),
            tone,
            reveal: None,
        });
    }

    fn open(&mut self, path: PathBuf) {
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        let (tx, rx) = channel();
        let p = path.clone();
        std::thread::spawn(move || {
            let _ = tx.send(Model::load(&p));
        });
        self.loading = Some(Loading {
            path,
            started: Instant::now(),
            rx,
        });
    }

    fn open_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new().add_filter("3D models", EXTENSIONS);
        if let Some(dir) = self
            .info
            .as_ref()
            .and_then(|i| i.path.parent())
            .or_else(|| self.p.recent.first().and_then(|r| r.parent()))
        {
            dialog = dialog.set_directory(dir);
        }
        if let Some(path) = dialog.pick_file() {
            self.open(path);
        }
    }

    fn poll_loading(&mut self) {
        let Some(loading) = &self.loading else {
            return;
        };
        let Ok(result) = loading.rx.try_recv() else {
            return;
        };
        let loading = self.loading.take().unwrap();
        match result {
            Ok(model) => {
                let info = ModelInfo {
                    path: loading.path.clone(),
                    name: model.file_name(),
                    triangles: model.triangle_count(),
                    vertices: model.vertex_count(),
                    materials: model.materials.len(),
                    textures: model.textures.len(),
                    warnings: model.warnings.clone(),
                    load_time: loading.started.elapsed(),
                };
                self.session.set_model(model);
                if let Some(p) = &mut self.preview {
                    p.rendered = None;
                }
                self.p.recent.retain(|r| r != &loading.path);
                self.p.recent.insert(0, loading.path);
                self.p.recent.truncate(MAX_RECENT);
                let warn = info.warnings.len();
                self.info = Some(info);
                if warn > 0 {
                    self.set_status(
                        format!(
                            "Loaded with {warn} warning{}",
                            if warn == 1 { "" } else { "s" }
                        ),
                        Tone::Bad,
                    );
                } else {
                    self.status = None;
                }
            }
            Err(e) => {
                self.set_status(
                    format!("Couldn't open {}: {e:#}", file_name(&loading.path)),
                    Tone::Bad,
                );
            }
        }
    }

    fn start_export(&mut self) {
        if self.export.is_some() {
            return;
        }
        let Some(info) = &self.info else {
            return;
        };
        let format = self.settings().output.format;
        let ffmpeg = match &self.ffmpeg {
            FfmpegState::Found(p) => Some(p.clone()),
            _ if format.needs_ffmpeg() => {
                self.set_status(
                    format!(
                        "{} needs ffmpeg, download it from the Output section",
                        format.label()
                    ),
                    Tone::Bad,
                );
                return;
            }
            _ => None,
        };
        let stem = info
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into());
        let dir = self
            .p
            .export_dir
            .clone()
            .filter(|d| d.is_dir())
            .or_else(|| info.path.parent().map(Path::to_path_buf));
        let mut dialog = rfd::FileDialog::new();
        if let Some(d) = &dir {
            dialog = dialog.set_directory(d);
        }
        let out = if format == OutputFormat::PngSequence {
            dialog
                .set_title("Folder for the PNG frames")
                .pick_folder()
                .map(|d| d.join(format!("{stem}_turntable")))
        } else {
            let ext = format.extension();
            dialog
                .set_file_name(format!("{stem}_turntable.{ext}"))
                .add_filter(format.label(), &[ext])
                .save_file()
                .map(|p| {
                    if p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext)) {
                        p
                    } else {
                        p.with_extension(ext)
                    }
                })
        };
        let Some(out) = out else {
            return;
        };
        self.p.export_dir = out.parent().map(Path::to_path_buf);
        let settings = self.p.settings.clone();
        let Some(job) = self.session.job(&settings, out, ffmpeg) else {
            return;
        };
        let progress = Arc::new(export::Progress::default());
        let p = progress.clone();
        let handle = std::thread::spawn(move || export::run(job, &p));
        self.status = None;
        self.export = Some(ExportState {
            progress,
            handle: Some(handle),
            started: Instant::now(),
        });
    }

    fn poll_export(&mut self) {
        let Some(state) = &mut self.export else {
            return;
        };
        if !state.handle.as_ref().is_some_and(|h| h.is_finished()) {
            return;
        }
        let elapsed = state.started.elapsed();
        let result = state
            .handle
            .take()
            .unwrap()
            .join()
            .unwrap_or_else(|_| Err(anyhow!("export thread panicked")));
        self.export = None;
        match result {
            Ok(path) => {
                let size = file_size(&path);
                self.status = Some(Status {
                    text: format!(
                        "Saved {}{} in {:.1}s",
                        file_name(&path),
                        size.map(|s| format!(" ({s})")).unwrap_or_default(),
                        elapsed.as_secs_f32()
                    ),
                    tone: Tone::Ok,
                    reveal: Some(path),
                });
            }
            Err(e) if e.to_string() == "Cancelled" => {
                self.set_status("Export cancelled", Tone::Info)
            }
            Err(e) => self.set_status(format!("Export failed: {e:#}"), Tone::Bad),
        }
    }

    fn poll_ffmpeg(&mut self) {
        if let FfmpegState::Downloading { rx, .. } = &self.ffmpeg
            && let Ok(result) = rx.try_recv()
        {
            match result {
                Ok(p) => {
                    self.ffmpeg = FfmpegState::Found(p);
                    self.set_status("ffmpeg is ready", Tone::Ok);
                }
                Err(e) => {
                    self.ffmpeg = FfmpegState::Missing;
                    self.set_status(format!("ffmpeg download failed: {e:#}"), Tone::Bad);
                }
            }
        }
    }

    fn download_ffmpeg(&mut self, ctx: &egui::Context) {
        let progress = Arc::new((AtomicU64::new(0), AtomicU64::new(0)));
        let (tx, rx) = channel();
        let p = progress.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let r = export::download_ffmpeg(|done, total| {
                p.0.store(done, Ordering::Relaxed);
                p.1.store(total, Ordering::Relaxed);
                ctx.request_repaint_after(Duration::from_millis(100));
            });
            let _ = tx.send(r);
            ctx.request_repaint();
        });
        self.ffmpeg = FfmpegState::Downloading { progress, rx };
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        if let Some(path) = dropped.into_iter().find(|p| is_model(p)) {
            self.open(path);
        }
        let (space, open, export) = ctx.input_mut(|i| {
            (
                i.consume_key(egui::Modifiers::NONE, egui::Key::Space),
                i.consume_key(egui::Modifiers::COMMAND, egui::Key::O),
                i.consume_key(egui::Modifiers::COMMAND, egui::Key::E),
            )
        });
        if space && !ctx.egui_wants_keyboard_input() {
            self.playing = !self.playing;
        }
        if open {
            self.open_dialog();
        }
        if export {
            self.start_export();
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_loading();
        self.poll_export();
        self.poll_ffmpeg();
        self.handle_input(&ctx);

        if self.playing && self.info.is_some() {
            let dt = ctx.input(|i| i.stable_dt).min(0.1);
            self.t = (self.t + dt / self.settings().motion.duration.max(0.1)).fract();
            ctx.request_repaint();
        }
        if self.loading.is_some() || self.export.is_some() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }

        egui::Panel::top("topbar")
            .frame(bar_frame(Margin::symmetric(8, 7)))
            .show(ui, |ui| self.topbar(ui));
        if self.export.is_some() {
            egui::Panel::top("jobbar")
                .frame(bar_frame(Margin::symmetric(12, 5)))
                .show(ui, |ui| self.jobbar(ui));
        }
        egui::Panel::bottom("statusbar")
            .frame(bar_frame(Margin::symmetric(12, 4)))
            .show(ui, |ui| self.statusbar(ui));
        egui::Panel::right("settings")
            .resizable(true)
            .default_size(350.0)
            .size_range(310.0..=520.0)
            .frame(Frame::new().fill(PANEL_2))
            .show(ui, |ui| self.settings_panel(ui));
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewport(ui));
        self.screenshot_hook(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.p);
    }
}

fn bar_frame(margin: Margin) -> Frame {
    Frame::new().fill(PANEL).inner_margin(margin)
}

impl App {
    fn screenshot_hook(&mut self, ctx: &egui::Context) {
        let Some(shot) = &mut self.screenshot else {
            return;
        };
        if shot.done.load(Ordering::Relaxed) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if self.info.is_none() && self.loading.is_some() {
            return;
        }
        shot.frames += 1;
        ctx.request_repaint();
        if shot.frames == 20 {
            let path = shot.path.clone();
            let done = shot.done.clone();
            ctx.request_screenshot(move |img| {
                let [w, h] = img.size;
                let px: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                if let Some(buf) = image::RgbaImage::from_raw(w as u32, h as u32, px) {
                    let _ = buf.save(&path);
                }
                done.store(true, Ordering::Relaxed);
            });
        }
    }

    fn topbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("model-turntable")
                    .family(semibold())
                    .size(FS_LG)
                    .color(FG),
            );
            ui.add_space(6.0);
            if ui.button("Open…").on_hover_text("Ctrl+O").clicked() {
                self.open_dialog();
            }
            let recent = self.p.recent.clone();
            ui.add_enabled_ui(!recent.is_empty(), |ui| {
                ui.menu_button("Recent", |ui| {
                    for r in &recent {
                        if ui
                            .button(file_name(r))
                            .on_hover_text(r.display().to_string())
                            .clicked()
                        {
                            self.open(r.clone());
                        }
                    }
                    ui.separator();
                    if ui.button("Clear list").clicked() {
                        self.p.recent.clear();
                    }
                });
            });
            ui.menu_button("Settings", |ui| {
                if ui.button("Save settings…").clicked() {
                    self.save_settings_dialog();
                }
                if ui.button("Load settings…").clicked() {
                    self.load_settings_dialog();
                }
                ui.separator();
                if ui.button("Reset to defaults").clicked() {
                    self.p.settings = Settings::default();
                }
            });
            ui.add_space(8.0);
            if let Some(info) = &self.info {
                ui.label(RichText::new(&info.name).family(semibold()).color(FG))
                    .on_hover_text(info.path.display().to_string());
            }
            if let Some(l) = &self.loading {
                ui.spinner();
                ui.label(RichText::new(format!("Loading {}…", file_name(&l.path))).color(FG_DIM));
            }

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let enabled = self.info.is_some() && self.export.is_none();
                let label = format!("Export {}", self.settings().output.format.label());
                let btn = Button::new(RichText::new(label).color(if enabled {
                    ACCENT
                } else {
                    FG_FAINT
                }))
                .stroke(Stroke::new(1.0, if enabled { ACCENT } else { LINE }));
                if ui
                    .add_enabled(enabled, btn)
                    .on_hover_text("Ctrl+E")
                    .clicked()
                {
                    self.start_export();
                }
                let (w, h) = self.settings().output.render_size();
                let frames = self
                    .settings()
                    .output
                    .frame_count(self.settings().motion.duration);
                ui.label(
                    RichText::new(format!(
                        "{w}×{h} · {} fps · {frames} frames",
                        self.settings().output.fps
                    ))
                    .color(FG_FAINT)
                    .size(FS_SM),
                );
            });
        });
    }

    fn jobbar(&mut self, ui: &mut Ui) {
        let Some(state) = &self.export else {
            return;
        };
        let progress = state.progress.clone();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(progress.stage().to_uppercase())
                    .size(FS_SM)
                    .extra_letter_spacing(1.0)
                    .color(ACCENT),
            );
            let done = progress.done.load(Ordering::Relaxed);
            let total = progress.total.load(Ordering::Relaxed);
            let cancel_w = 70.0;
            let count = format!("{done} / {total}");
            let bar_w = (ui.available_width() - cancel_w - 90.0).max(40.0);
            progress_bar(ui, bar_w, progress.fraction());
            ui.label(RichText::new(count).size(FS_SM).color(FG_DIM));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let btn =
                    Button::new(RichText::new("Cancel").color(SEL)).stroke(Stroke::new(1.0, SEL));
                if ui.add(btn).clicked() {
                    progress.cancel.store(true, Ordering::Relaxed);
                }
            });
        });
    }

    fn statusbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 12.0;
            if let Some(info) = &self.info {
                let stats = format!(
                    "{} tris · {} verts · {} material{} · {} texture{}",
                    thousands(info.triangles),
                    thousands(info.vertices),
                    info.materials,
                    plural(info.materials),
                    info.textures,
                    plural(info.textures),
                );
                ui.label(RichText::new(stats).size(FS_SM).color(FG_DIM))
                    .on_hover_text(format!("Loaded in {:.2}s", info.load_time.as_secs_f32()));
                if !info.warnings.is_empty() {
                    let n = info.warnings.len();
                    ui.label(
                        RichText::new(format!("{n} warning{}", plural(n)))
                            .size(FS_SM)
                            .color(SEL),
                    )
                    .on_hover_ui(|ui| {
                        for w in &info.warnings {
                            ui.label(RichText::new(w).size(FS_SM));
                        }
                    });
                }
            }
            let mut reveal = None;
            if let Some(s) = &self.status {
                let color = match s.tone {
                    Tone::Info => FG_DIM,
                    Tone::Ok => OK,
                    Tone::Bad => BAD,
                };
                ui.label(RichText::new(&s.text).size(FS_SM).color(color));
                if let Some(path) = &s.reveal
                    && ui
                        .add(
                            Button::new(
                                RichText::new("Show in folder")
                                    .size(FS_SM)
                                    .underline()
                                    .color(FG_DIM),
                            )
                            .frame(false),
                        )
                        .clicked()
                {
                    reveal = Some(path.clone());
                }
            }
            if let Some(p) = reveal {
                reveal_in_folder(&p);
            }
        });
    }

    fn save_settings_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("turntable-settings.json")
            .add_filter("JSON", &["json"])
            .save_file()
        {
            let r = serde_json::to_string_pretty(&self.p.settings)
                .map_err(anyhow::Error::from)
                .and_then(|s| Ok(std::fs::write(&path, s)?));
            match r {
                Ok(()) => self.set_status(format!("Saved {}", file_name(&path)), Tone::Ok),
                Err(e) => self.set_status(format!("Couldn't save settings: {e}"), Tone::Bad),
            }
        }
    }

    fn load_settings_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .pick_file()
        {
            let r = std::fs::read_to_string(&path)
                .map_err(anyhow::Error::from)
                .and_then(|s| Ok(serde_json::from_str::<Settings>(&s)?));
            match r {
                Ok(s) => {
                    self.p.settings = s;
                    self.set_status(format!("Loaded {}", file_name(&path)), Tone::Ok);
                }
                Err(e) => self.set_status(format!("Couldn't load settings: {e}"), Tone::Bad),
            }
        }
    }
}

impl App {
    fn viewport(&mut self, ui: &mut Ui) {
        let full = ui.available_rect_before_wrap();
        let transport_h = 36.0;
        let view = Rect::from_min_max(full.min, pos2(full.max.x, full.max.y - transport_h));
        let transport = Rect::from_min_max(pos2(full.min.x, view.max.y), full.max);
        ui.painter().rect_filled(view, 0.0, BG);

        let hovering_files = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());

        if self.info.is_some() {
            self.draw_preview(ui, view);
        } else {
            self.empty_state(ui, view);
        }

        if hovering_files {
            let r = view.shrink(12.0);
            ui.painter().rect_filled(
                r,
                CornerRadius::same(RADIUS),
                Color32::from_rgba_unmultiplied(0x1d, 0x29, 0x37, 220),
            );
            ui.painter().rect_stroke(
                r,
                CornerRadius::same(RADIUS),
                Stroke::new(2.0, ACCENT),
                StrokeKind::Inside,
            );
            ui.painter().text(
                r.center(),
                Align2::CENTER_CENTER,
                "Drop to load",
                FontId::new(FS_LG, semibold()),
                ACCENT,
            );
        }

        let mut tui = ui.new_child(
            UiBuilder::new()
                .max_rect(transport)
                .layout(Layout::left_to_right(Align::Center)),
        );
        tui.painter().rect_filled(transport, 0.0, PANEL);
        tui.painter().hline(
            transport.x_range(),
            transport.min.y + 0.5,
            Stroke::new(1.0, LINE),
        );
        self.transport(&mut tui, transport);
    }

    fn transport(&mut self, ui: &mut Ui, rect: Rect) {
        ui.add_space(8.0);
        ui.set_clip_rect(rect);
        let enabled = self.info.is_some();
        ui.add_enabled_ui(enabled, |ui| {
            let resp = ui
                .add(Button::new("").min_size(vec2(30.0, 24.0)))
                .on_hover_text("Play / pause (Space)");
            paint_play_icon(
                ui.painter(),
                resp.rect.center(),
                self.playing,
                if enabled { FG } else { FG_FAINT },
            );
            if resp.clicked() {
                self.playing = !self.playing;
            }
            let s = &self.p.settings;
            let frames = s.output.frame_count(s.motion.duration);
            let mut frame = (self.t * frames as f32).floor() as u32;
            ui.spacing_mut().slider_width = (rect.width() - 260.0).max(80.0);
            let resp = ui.add(
                egui::Slider::new(&mut frame, 0..=frames - 1)
                    .show_value(false)
                    .trailing_fill(true),
            );
            if resp.changed() {
                self.playing = false;
                self.t = frame as f32 / frames as f32;
            }
            let angle = s.motion.angle_at(self.t).rem_euclid(360.0);
            ui.label(
                RichText::new(format!("{:>4} / {frames}  ·  {angle:>3.0}°", frame + 1))
                    .monospace()
                    .size(FS_SM)
                    .color(FG_DIM),
            );
        });
    }

    fn empty_state(&mut self, ui: &mut Ui, view: Rect) {
        let mut child = ui.new_child(
            UiBuilder::new()
                .max_rect(view)
                .layout(Layout::top_down(Align::Center)),
        );
        let ui = &mut child;
        ui.add_space((view.height() * 0.28).max(40.0));
        ui.label(
            RichText::new("Drop a model here")
                .size(18.0)
                .family(semibold())
                .color(FG_DIM),
        );
        ui.label(RichText::new("glb · gltf · fbx · obj · stl").color(FG_FAINT));
        ui.add_space(10.0);
        if ui
            .add(
                Button::new(RichText::new("Open model…").color(ACCENT))
                    .stroke(Stroke::new(1.0, ACCENT)),
            )
            .clicked()
        {
            self.open_dialog();
        }
    }

    fn draw_preview(&mut self, ui: &mut Ui, view: Rect) {
        let (ow, oh) = self.settings().output.render_size();
        let aspect = ow as f32 / oh as f32;
        let area = view.shrink(28.0);
        let mut size = vec2(area.width(), area.width() / aspect);
        if size.y > area.height() {
            size = vec2(area.height() * aspect, area.height());
        }
        let frame_rect = Rect::from_center_size(area.center(), size.max(vec2(16.0, 16.0)));
        let ppp = ui.ctx().pixels_per_point();
        let px = (
            ((frame_rect.width() * ppp).round() as u32).max(16),
            ((frame_rect.height() * ppp).round() as u32).max(16),
        );

        let resp = ui.interact(
            view,
            ui.scope_id().with("viewport"),
            Sense::click_and_drag(),
        );
        if resp.dragged() {
            self.playing = false;
            let dx = resp.drag_delta().x / frame_rect.width().max(1.0);
            let s = &self.p.settings.motion;
            let span = match s.kind {
                MotionKind::Spin => 360.0 * s.turns.max(1) as f32,
                MotionKind::Swing => 360.0,
            };
            let dir = if s.clockwise { -1.0 } else { 1.0 };
            self.t = (self.t + dir * dx * 180.0 / span).rem_euclid(1.0);
        }
        if resp.hovered() {
            let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let z = &mut self.p.settings.camera.zoom;
                *z = (*z * (scroll * 0.002).exp()).clamp(0.2, 5.0);
            }
        }
        if resp.double_clicked() {
            self.p.settings.camera.zoom = 1.0;
        }

        let transparent = self.settings().background.transparent;
        if let Some(texture) = self.render_preview(px, transparent) {
            if transparent {
                checkerboard(ui.painter(), frame_rect);
            }
            ui.painter().image(
                texture,
                frame_rect,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        ui.painter()
            .rect_stroke(frame_rect, 0.0, Stroke::new(1.0, LINE), StrokeKind::Outside);
        ui.painter().text(
            frame_rect.left_top() + vec2(0.0, -6.0),
            Align2::LEFT_BOTTOM,
            format!("{ow} × {oh}"),
            FontId::proportional(FS_SM),
            FG_FAINT,
        );
        ui.painter().text(
            frame_rect.right_top() + vec2(0.0, -6.0),
            Align2::RIGHT_BOTTOM,
            "drag to scrub · scroll to zoom",
            FontId::proportional(FS_SM),
            FG_FAINT,
        );
    }

    fn render_preview(&mut self, px: (u32, u32), transparent: bool) -> Option<egui::TextureId> {
        let settings = self.p.settings.clone();
        let angle = settings.motion.angle_at(self.t);
        let renderer = self.session.renderer.clone();

        let needs_new = self
            .preview
            .as_ref()
            .is_none_or(|p| (p.target.width, p.target.height) != px);
        if needs_new {
            let target = renderer.create_target(px.0, px.1, 1);
            let mut egui_renderer = self.render_state.renderer.write();
            let texture = match &self.preview {
                Some(old) => {
                    egui_renderer.update_egui_texture_from_wgpu_texture(
                        &self.render_state.device,
                        target.display_view(),
                        wgpu::FilterMode::Linear,
                        old.texture,
                    );
                    old.texture
                }
                None => egui_renderer.register_native_texture(
                    &self.render_state.device,
                    target.display_view(),
                    wgpu::FilterMode::Linear,
                ),
            };
            self.preview = Some(Preview {
                target,
                texture,
                rendered: None,
            });
        }

        let key = (settings.clone(), angle, transparent);
        let preview = self.preview.as_ref()?;
        if preview.rendered.as_ref() == Some(&key) {
            return Some(preview.texture);
        }
        let prepared = self.session.prepare(&settings)?;
        let uniforms = prepared.uniforms(&settings, px.0 as f32 / px.1 as f32, angle, transparent);
        let preview = self.preview.as_mut()?;
        renderer.render(
            &preview.target,
            &prepared.gpu,
            &prepared.resources,
            &uniforms,
            !transparent,
            settings.background.ground_shadow,
        );
        preview.rendered = Some(key);
        Some(preview.texture)
    }
}

impl App {
    fn settings_panel(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                self.section(ui, "model", "Model", Self::model_section);
                self.section(ui, "lighting", "Lighting", Self::lighting_section);
                self.section(ui, "background", "Background", Self::background_section);
                self.section(ui, "camera", "Camera", Self::camera_section);
                self.section(ui, "motion", "Motion", Self::motion_section);
                self.section(ui, "output", "Output", Self::output_section);
            });
    }

    fn section(&mut self, ui: &mut Ui, id: &str, title: &str, body: fn(&mut Self, &mut Ui)) {
        let open = !self.p.closed_sections.contains(id);
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 28.0), Sense::click());
        let painter = ui.painter();
        painter.rect_filled(rect, 0.0, if resp.hovered() { ROW_HOVER } else { PANEL });
        painter.hline(rect.x_range(), rect.max.y - 0.5, Stroke::new(1.0, LINE));
        if rect.min.y > ui.min_rect().min.y + 1.0 {
            painter.hline(rect.x_range(), rect.min.y + 0.5, Stroke::new(1.0, LINE));
        }
        ui.painter().text(
            rect.left_center() + vec2(12.0, 0.0),
            Align2::LEFT_CENTER,
            spaced(&title.to_uppercase()),
            FontId::proportional(FS_SM),
            if resp.hovered() { FG_DIM } else { FG_FAINT },
        );
        paint_chevron(
            ui.painter(),
            rect.right_center() - vec2(16.0, 0.0),
            open,
            FG_FAINT,
        );
        if resp.clicked() {
            if open {
                self.p.closed_sections.insert(id.to_owned());
            } else {
                self.p.closed_sections.remove(id);
            }
        }
        if open {
            Frame::new()
                .inner_margin(Margin {
                    left: 12,
                    right: 12,
                    top: 10,
                    bottom: 12,
                })
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 7.0;
                    ui.set_width(ui.available_width());
                    body(self, ui);
                });
        }
    }

    fn model_section(&mut self, ui: &mut Ui) {
        let m = &mut self.p.settings.model;
        row(ui, "Up axis", |ui| {
            segmented(
                ui,
                &mut m.up,
                &[
                    (UpAxis::Auto, "Auto"),
                    (UpAxis::Y, "Y up"),
                    (UpAxis::Z, "Z up"),
                ],
            );
        });
        for (i, axis) in ["Rotate X", "Rotate Y", "Rotate Z"].iter().enumerate() {
            row(ui, axis, |ui| {
                ui.add(
                    egui::DragValue::new(&mut m.rotation[i])
                        .speed(1.0)
                        .range(-180.0..=180.0)
                        .suffix("°")
                        .fixed_decimals(0),
                );
                for step in [-90.0, 90.0] {
                    let label = if step < 0.0 { "−90" } else { "+90" };
                    if small_button(ui, label).clicked() {
                        m.rotation[i] = wrap_degrees(m.rotation[i] + step);
                    }
                }
                if m.rotation[i] != 0.0 && small_button(ui, "Reset").clicked() {
                    m.rotation[i] = 0.0;
                }
            });
        }
        row(ui, "Shading", |ui| {
            segmented(
                ui,
                &mut m.shading,
                &[
                    (Shading::Auto, "Auto"),
                    (Shading::Smooth, "Smooth"),
                    (Shading::Flat, "Flat"),
                ],
            );
        });
        row(ui, "Clay", |ui| {
            ui.checkbox(&mut m.clay, "Ignore materials");
            if m.clay {
                ui.color_edit_button_srgb(&mut m.clay_color);
            }
        });
        if m.clay {
            row(ui, "Roughness", |ui| {
                ui.add(egui::Slider::new(&mut m.clay_roughness, 0.05..=1.0).fixed_decimals(2));
            });
        }
    }

    fn lighting_section(&mut self, ui: &mut Ui) {
        let l = &mut self.p.settings.lighting;
        row(ui, "Mode", |ui| {
            segmented(
                ui,
                &mut l.mode,
                &[
                    (LightingMode::Studio, "Studio HDRI"),
                    (LightingMode::Lights, "Lights"),
                    (LightingMode::Matcap, "MatCap"),
                ],
            );
        });
        match l.mode {
            LightingMode::Studio => {
                ui.label(
                    RichText::new("Image-based lighting, like Blender's Material Preview.")
                        .size(FS_SM)
                        .color(FG_FAINT),
                );
                let selected = l.hdri.clone();
                if let Some(id) = thumb_grid(
                    ui,
                    &self.hdri_thumbs,
                    HDRIS.iter().map(|h| (h.id, h.label)),
                    &selected,
                    2.0,
                    3,
                ) {
                    l.hdri = id.to_owned();
                }
                row(ui, "Rotation", |ui| {
                    ui.add(
                        egui::Slider::new(&mut l.hdri_rotation, 0.0..=360.0)
                            .suffix("°")
                            .fixed_decimals(0),
                    );
                });
                row(ui, "Strength", |ui| {
                    ui.add(egui::Slider::new(&mut l.hdri_intensity, 0.0..=4.0).fixed_decimals(2));
                });
            }
            LightingMode::Lights => {
                row(ui, "Preset", |ui| {
                    egui::ComboBox::from_id_salt("preset")
                        .width(ui.available_width())
                        .selected_text(&l.preset)
                        .show_ui(ui, |ui| {
                            for p in PRESETS {
                                if ui.selectable_label(l.preset == p.name, p.name).clicked() {
                                    l.preset = p.name.to_owned();
                                    l.follow_model = p.follow_model;
                                }
                            }
                        });
                });
                row(ui, "Intensity", |ui| {
                    ui.add(egui::Slider::new(&mut l.preset_intensity, 0.0..=3.0).fixed_decimals(2));
                });
            }
            LightingMode::Matcap => {
                let selected = l.matcap.clone();
                if let Some(id) = thumb_grid(
                    ui,
                    &self.matcap_thumbs,
                    MATCAPS.iter().map(|m| (*m, *m)),
                    &selected,
                    1.0,
                    6,
                ) {
                    l.matcap = id.to_owned();
                }
                row(ui, "", |ui| {
                    ui.checkbox(&mut l.matcap_use_colors, "Tint with model colours");
                });
            }
        }
        if l.mode != LightingMode::Matcap {
            row(ui, "", |ui| {
                ui.checkbox(&mut l.follow_model, "Lighting spins with model")
                    .on_hover_text("Off: the model turns under fixed lights, so highlights travel across it.\nOn: the lighting on the model never changes.");
            });
        }
        row(ui, "Exposure", |ui| {
            ui.add(
                egui::Slider::new(&mut l.exposure, -3.0..=3.0)
                    .suffix(" EV")
                    .fixed_decimals(1),
            );
        });
        row(ui, "Tonemap", |ui| {
            let opts: Vec<(Tonemap, &str)> = Tonemap::ALL.iter().map(|t| (*t, t.label())).collect();
            segmented(ui, &mut l.tonemap, &opts);
        });
    }

    fn background_section(&mut self, ui: &mut Ui) {
        let b = &mut self.p.settings.background;
        let format = self.p.settings.output.format;
        row(ui, "", |ui| {
            ui.checkbox(&mut b.transparent, "Transparent");
        });
        if b.transparent && !format.supports_alpha() {
            ui.label(
                RichText::new(format!(
                    "{} has no alpha, the background will be filled.",
                    format.label()
                ))
                .size(FS_SM)
                .color(SEL),
            );
        }
        if !b.transparent || !format.supports_alpha() {
            row(ui, "Style", |ui| {
                let opts: Vec<(BackgroundStyle, &str)> = [
                    BackgroundStyle::Solid,
                    BackgroundStyle::Vertical,
                    BackgroundStyle::Radial,
                    BackgroundStyle::Hdri,
                ]
                .iter()
                .map(|s| (*s, s.label()))
                .collect();
                segmented(ui, &mut b.style, &opts);
            });
            match b.style {
                BackgroundStyle::Solid => row(ui, "Colour", |ui| {
                    ui.color_edit_button_srgb(&mut b.color);
                }),
                BackgroundStyle::Vertical => row(ui, "Top / bottom", |ui| {
                    ui.color_edit_button_srgb(&mut b.color);
                    ui.color_edit_button_srgb(&mut b.color2);
                    if small_button(ui, "Swap").clicked() {
                        std::mem::swap(&mut b.color, &mut b.color2);
                    }
                }),
                BackgroundStyle::Radial => row(ui, "Centre / edge", |ui| {
                    ui.color_edit_button_srgb(&mut b.color);
                    ui.color_edit_button_srgb(&mut b.color2);
                    if small_button(ui, "Swap").clicked() {
                        std::mem::swap(&mut b.color, &mut b.color2);
                    }
                }),
                BackgroundStyle::Hdri => row(ui, "Blur", |ui| {
                    ui.add(egui::Slider::new(&mut b.hdri_blur, 0.0..=1.0).fixed_decimals(2));
                }),
            };
        }
        if format == OutputFormat::Gif && b.transparent {
            row(ui, "Edge matte", |ui| {
                ui.color_edit_button_srgb(&mut b.color);
                ui.label(
                    RichText::new("GIF alpha is on/off, soft edges blend into this")
                        .size(FS_SM)
                        .color(FG_FAINT),
                );
            });
        }
        row(ui, "", |ui| {
            ui.checkbox(&mut b.ground_shadow, "Ground shadow");
        });
        if b.ground_shadow {
            row(ui, "Shadow", |ui| {
                ui.add(egui::Slider::new(&mut b.shadow_strength, 0.0..=1.0).fixed_decimals(2));
            });
        }
    }

    fn camera_section(&mut self, ui: &mut Ui) {
        let c = &mut self.p.settings.camera;
        row(ui, "Elevation", |ui| {
            ui.add(
                egui::Slider::new(&mut c.elevation, -45.0..=85.0)
                    .suffix("°")
                    .fixed_decimals(0),
            );
        });
        row(ui, "Zoom", |ui| {
            ui.add(
                egui::Slider::new(&mut c.zoom, 0.2..=5.0)
                    .logarithmic(true)
                    .fixed_decimals(2),
            );
        });
        row(ui, "Field of view", |ui| {
            ui.add(
                egui::Slider::new(&mut c.fov, 10.0..=90.0)
                    .suffix("°")
                    .fixed_decimals(0),
            );
        });
        row(ui, "Vertical shift", |ui| {
            ui.add(egui::Slider::new(&mut c.offset_y, -0.5..=0.5).fixed_decimals(2));
        });
        if small_button(ui, "Reset camera").clicked() {
            *c = CameraSettings::default();
        }
    }

    fn motion_section(&mut self, ui: &mut Ui) {
        let m = &mut self.p.settings.motion;
        row(ui, "Motion", |ui| {
            segmented(
                ui,
                &mut m.kind,
                &[(MotionKind::Spin, "Spin"), (MotionKind::Swing, "Swing")],
            );
        });
        row(ui, "Loop length", |ui| {
            ui.add(
                egui::DragValue::new(&mut m.duration)
                    .speed(0.1)
                    .range(0.5..=120.0)
                    .suffix(" s")
                    .fixed_decimals(1),
            );
            for d in [3.0, 6.0, 10.0] {
                if small_button(ui, &format!("{d:.0}s")).clicked() {
                    m.duration = d;
                }
            }
        });
        row(ui, "Direction", |ui| {
            segmented(
                ui,
                &mut m.clockwise,
                &[(false, "Counter-clockwise"), (true, "Clockwise")],
            );
        });
        match m.kind {
            MotionKind::Spin => {
                row(ui, "Turns", |ui| {
                    ui.add(egui::DragValue::new(&mut m.turns).range(1..=10).speed(0.05));
                    ui.checkbox(&mut m.ease, "Ease in/out");
                });
            }
            MotionKind::Swing => {
                row(ui, "Swing", |ui| {
                    ui.add(
                        egui::Slider::new(&mut m.swing_angle, 5.0..=180.0)
                            .suffix("°")
                            .fixed_decimals(0),
                    );
                });
            }
        }
        row(ui, "Start angle", |ui| {
            ui.add(
                egui::Slider::new(&mut m.start_angle, -180.0..=180.0)
                    .suffix("°")
                    .fixed_decimals(0),
            );
        });
    }

    fn output_section(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let o = &mut self.p.settings.output;
        row(ui, "Format", |ui| {
            egui::ComboBox::from_id_salt("format")
                .width(ui.available_width())
                .selected_text(o.format.label())
                .show_ui(ui, |ui| {
                    for f in OutputFormat::ALL {
                        ui.selectable_value(&mut o.format, f, f.label());
                    }
                });
        });
        row(ui, "Size", |ui| {
            let current = SIZE_PRESETS
                .iter()
                .find(|(_, w, h)| *w == o.width && *h == o.height)
                .map(|(n, _, _)| *n)
                .unwrap_or("Custom");
            egui::ComboBox::from_id_salt("size")
                .width(ui.available_width())
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (name, w, h) in SIZE_PRESETS {
                        if ui
                            .selectable_label(o.width == *w && o.height == *h, *name)
                            .clicked()
                        {
                            o.width = *w;
                            o.height = *h;
                        }
                    }
                });
        });
        row(ui, "", |ui| {
            ui.add(
                egui::DragValue::new(&mut o.width)
                    .range(16..=7680)
                    .speed(2.0)
                    .suffix(" px"),
            );
            ui.label(RichText::new("×").color(FG_FAINT));
            ui.add(
                egui::DragValue::new(&mut o.height)
                    .range(16..=7680)
                    .speed(2.0)
                    .suffix(" px"),
            );
            if small_button(ui, "Swap")
                .on_hover_text("Swap width and height")
                .clicked()
            {
                std::mem::swap(&mut o.width, &mut o.height);
            }
        });
        let (rw, rh) = o.render_size();
        if (rw, rh) != (o.width, o.height) {
            ui.label(
                RichText::new(format!(
                    "{} needs even sizes, renders at {rw}×{rh}",
                    o.format.label()
                ))
                .size(FS_SM)
                .color(FG_FAINT),
            );
        }
        row(ui, "Frame rate", |ui| {
            ui.add(
                egui::DragValue::new(&mut o.fps)
                    .range(1..=120)
                    .suffix(" fps"),
            );
            for f in [24, 30, 50, 60] {
                if small_button(ui, &f.to_string()).clicked() {
                    o.fps = f;
                }
            }
        });
        if o.format == OutputFormat::Gif && o.fps > 50 {
            ui.label(
                RichText::new("GIF timing tops out at 50 fps, faster plays back slower.")
                    .size(FS_SM)
                    .color(SEL),
            );
        }
        if o.format.has_quality() {
            row(ui, "Quality", |ui| {
                ui.add(egui::Slider::new(&mut o.quality, 1..=100));
            });
        }
        row(ui, "Anti-alias", |ui| {
            segmented(
                ui,
                &mut o.supersample,
                &[(1, "1×"), (2, "2×"), (3, "3×"), (4, "4×")],
            );
        })
        .on_hover_text("Renders larger and scales down, on top of 4× MSAA");
        if o.format == OutputFormat::Gif {
            row(ui, "", |ui| {
                ui.checkbox(&mut o.gif_dither, "Dither")
                    .on_hover_text("Smoother gradients, slightly noisier and larger files");
            });
        }

        if o.format.needs_ffmpeg() {
            ui.add_space(2.0);
            match &self.ffmpeg {
                FfmpegState::Found(p) => {
                    ui.label(RichText::new("ffmpeg ready").size(FS_SM).color(FG_FAINT))
                        .on_hover_text(p.display().to_string());
                }
                FfmpegState::Missing => {
                    ui.label(
                        RichText::new("This format needs ffmpeg, which wasn't found.")
                            .size(FS_SM)
                            .color(SEL),
                    );
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                Button::new(RichText::new("Download ffmpeg").color(ACCENT))
                                    .stroke(Stroke::new(1.0, ACCENT)),
                            )
                            .clicked()
                        {
                            self.download_ffmpeg(&ctx);
                        }
                        if cfg!(target_os = "linux") {
                            ui.label(
                                RichText::new("or install it with your package manager")
                                    .size(FS_SM)
                                    .color(FG_FAINT),
                            );
                        }
                    });
                }
                FfmpegState::Downloading { progress, .. } => {
                    let done = progress.0.load(Ordering::Relaxed);
                    let total = progress.1.load(Ordering::Relaxed).max(1);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("Downloading ffmpeg")
                                .size(FS_SM)
                                .color(FG_DIM),
                        );
                        progress_bar(ui, 120.0, done as f32 / total as f32);
                        ui.label(
                            RichText::new(format!("{} MB", done / 1_000_000))
                                .size(FS_SM)
                                .color(FG_FAINT),
                        );
                    });
                }
            }
        }
    }
}

fn row(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui)) -> egui::Response {
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            vec2(LABEL_W, 22.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.set_min_width(LABEL_W);
                ui.label(RichText::new(label).color(FG_DIM));
            },
        );
        add(ui);
    })
    .response
}

fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, options: &[(T, &str)]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.spacing_mut().button_padding = vec2(8.0, 3.0);
        let n = options.len();
        for (i, (v, label)) in options.iter().enumerate() {
            let on = *value == *v;
            let r = RADIUS;
            let corner = CornerRadius {
                nw: if i == 0 { r } else { 0 },
                sw: if i == 0 { r } else { 0 },
                ne: if i == n - 1 { r } else { 0 },
                se: if i == n - 1 { r } else { 0 },
            };
            let btn = Button::new(RichText::new(*label).size(12.0).color(if on {
                ACCENT
            } else {
                FG_DIM
            }))
            .fill(if on { ROW_SEL } else { PANEL_2 })
            .stroke(Stroke::new(1.0, if on { ACCENT_DIM } else { LINE }))
            .corner_radius(corner);
            if ui.add(btn).clicked() && !on {
                *value = *v;
                changed = true;
            }
        }
    });
    changed
}

fn small_button(ui: &mut Ui, label: &str) -> egui::Response {
    ui.scope(|ui| {
        ui.spacing_mut().button_padding = vec2(6.0, 1.0);
        ui.add(Button::new(RichText::new(label).size(FS_SM).color(FG_DIM)))
    })
    .inner
}

fn thumb_grid<'a>(
    ui: &mut Ui,
    thumbs: &[TextureHandle],
    items: impl Iterator<Item = (&'a str, &'a str)>,
    selected: &str,
    aspect: f32,
    columns: usize,
) -> Option<&'a str> {
    let gap = 6.0;
    let width = ui.available_width();
    let w = ((width - gap * (columns - 1) as f32) / columns as f32).floor();
    let h = (w / aspect).floor();
    let label_h = if aspect > 1.0 { 16.0 } else { 15.0 };
    let mut picked = None;
    let items: Vec<_> = items.collect();
    for chunk in items.chunks(columns).zip(thumbs.chunks(columns)) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for ((id, label), tex) in chunk.0.iter().zip(chunk.1) {
                let (rect, resp) = ui.allocate_exact_size(vec2(w, h + label_h), Sense::click());
                let img = Rect::from_min_size(rect.min, vec2(w, h));
                let round = if aspect > 1.0 {
                    CornerRadius::same(4)
                } else {
                    CornerRadius::same((w / 2.0) as u8)
                };
                let p = ui.painter();
                if aspect > 1.0 {
                    p.add(
                        egui::epaint::RectShape::filled(img, round, Color32::WHITE).with_texture(
                            tex.id(),
                            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                        ),
                    );
                } else {
                    p.image(
                        tex.id(),
                        img,
                        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }
                let on = *id == selected;
                let stroke = if on {
                    Stroke::new(2.0, ACCENT)
                } else if resp.hovered() {
                    Stroke::new(1.0, ACCENT_DIM)
                } else {
                    Stroke::new(1.0, LINE)
                };
                p.rect_stroke(img, round, stroke, StrokeKind::Outside);
                p.text(
                    pos2(img.center().x, img.max.y + 3.0),
                    Align2::CENTER_TOP,
                    *label,
                    FontId::proportional(10.5),
                    if on { ACCENT } else { FG_DIM },
                );
                if resp.clicked() {
                    picked = Some(*id);
                }
            }
        });
    }
    picked
}

fn paint_play_icon(p: &egui::Painter, c: egui::Pos2, playing: bool, color: Color32) {
    if playing {
        for dx in [-3.5, 3.5] {
            p.rect_filled(
                Rect::from_center_size(c + vec2(dx, 0.0), vec2(3.0, 11.0)),
                1.0,
                color,
            );
        }
    } else {
        let pts = vec![
            c + vec2(-4.0, -6.0),
            c + vec2(6.0, 0.0),
            c + vec2(-4.0, 6.0),
        ];
        p.add(egui::Shape::convex_polygon(pts, color, Stroke::NONE));
    }
}

fn paint_chevron(p: &egui::Painter, c: egui::Pos2, open: bool, color: Color32) {
    let pts = if open {
        vec![
            c + vec2(-4.0, -2.0),
            c + vec2(0.0, 2.0),
            c + vec2(4.0, -2.0),
        ]
    } else {
        vec![
            c + vec2(-2.0, -4.0),
            c + vec2(2.0, 0.0),
            c + vec2(-2.0, 4.0),
        ]
    };
    p.add(egui::Shape::line(pts, Stroke::new(1.5, color)));
}

fn progress_bar(ui: &mut Ui, width: f32, fraction: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(width, 4.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, PANEL_2);
    let mut fill = rect;
    fill.set_width(rect.width() * fraction.clamp(0.0, 1.0));
    ui.painter().rect_filled(fill, 2.0, ACCENT);
}

fn checkerboard(p: &egui::Painter, rect: Rect) {
    let s = 10.0;
    p.rect_filled(rect, 0.0, Color32::from_gray(48));
    let cols = (rect.width() / s).ceil() as i32;
    let rows = (rect.height() / s).ceil() as i32;
    for y in 0..rows {
        for x in 0..cols {
            if (x + y) % 2 == 0 {
                let min = rect.min + vec2(x as f32 * s, y as f32 * s);
                let r = Rect::from_min_size(min, vec2(s, s)).intersect(rect);
                p.rect_filled(r, 0.0, Color32::from_gray(64));
            }
        }
    }
}

fn hdri_thumbnail(img: &HdrImage, w: u32, h: u32) -> egui::ColorImage {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = (x * img.width / w).min(img.width - 1);
            let sy = (y * img.height / h).min(img.height - 1);
            let c = img.pixels[(sy * img.width + sx) as usize];
            for v in [c.x, c.y, c.z] {
                let t = v / (1.0 + v);
                px.push(export::linear_to_srgb(t * 1.4));
            }
            px.push(255);
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px)
}

fn matcap_thumbnail(data: &[half::f16]) -> egui::ColorImage {
    let s = crate::render::env::MATCAP_SIZE as usize;
    let out = 96usize;
    let mut px = Vec::with_capacity(out * out * 4);
    for y in 0..out {
        for x in 0..out {
            let sx = x * s / out;
            let sy = y * s / out;
            let i = (sy * s + sx) * 4;
            let nx = (x as f32 + 0.5) / out as f32 * 2.0 - 1.0;
            let ny = (y as f32 + 0.5) / out as f32 * 2.0 - 1.0;
            let a = ((1.0 - (nx * nx + ny * ny).sqrt()) * out as f32 * 0.5).clamp(0.0, 1.0);
            for c in 0..3 {
                let v = data[i + c].to_f32();
                px.push(export::linear_to_srgb(v / (1.0 + v * 0.25)));
            }
            px.push((a * 255.0) as u8);
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([out, out], &px)
}

fn is_model(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

fn file_size(p: &Path) -> Option<String> {
    let meta = std::fs::metadata(p).ok()?;
    if meta.is_dir() {
        return None;
    }
    let b = meta.len() as f64;
    Some(if b > 1_000_000.0 {
        format!("{:.1} MB", b / 1_000_000.0)
    } else {
        format!("{:.0} KB", b / 1000.0)
    })
}

fn reveal_in_folder(p: &Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer")
            .arg(format!("/select,{}", p.display()))
            .spawn();
    }
    #[cfg(not(windows))]
    {
        let dir = if p.is_dir() {
            p
        } else {
            p.parent().unwrap_or(p)
        };
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn spaced(s: &str) -> String {
    s.chars()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("\u{200A}")
}

fn wrap_degrees(v: f32) -> f32 {
    let w = (v + 180.0).rem_euclid(360.0) - 180.0;
    if w == -180.0 { 180.0 } else { w }
}
