use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Settings {
    pub model: ModelSettings,
    pub camera: CameraSettings,
    pub lighting: LightingSettings,
    pub background: BackgroundSettings,
    pub motion: MotionSettings,
    pub output: OutputSettings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpAxis {
    Auto,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shading {
    Auto,
    Smooth,
    Flat,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSettings {
    pub up: UpAxis,
    pub rotation: [f32; 3],
    pub shading: Shading,
    pub clay: bool,
    pub clay_color: [u8; 3],
    pub clay_roughness: f32,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            up: UpAxis::Auto,
            rotation: [0.0; 3],
            shading: Shading::Auto,
            clay: false,
            clay_color: [200, 200, 200],
            clay_roughness: 0.55,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraSettings {
    pub elevation: f32,
    pub fov: f32,
    pub zoom: f32,
    pub offset_y: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            elevation: 15.0,
            fov: 30.0,
            zoom: 1.0,
            offset_y: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingMode {
    Studio,
    Lights,
    Matcap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tonemap {
    AgX,
    Aces,
    Standard,
}

impl Tonemap {
    pub const ALL: [Tonemap; 3] = [Tonemap::AgX, Tonemap::Aces, Tonemap::Standard];

    pub fn label(self) -> &'static str {
        match self {
            Tonemap::AgX => "AgX",
            Tonemap::Aces => "ACES",
            Tonemap::Standard => "Standard",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LightingSettings {
    pub mode: LightingMode,
    pub hdri: String,
    pub hdri_rotation: f32,
    pub hdri_intensity: f32,
    pub preset: String,
    pub preset_intensity: f32,
    pub matcap: String,
    pub matcap_use_colors: bool,
    /// When set, lights (or the HDRI) spin with the model so the lighting on it never changes.
    pub follow_model: bool,
    pub exposure: f32,
    pub tonemap: Tonemap,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            mode: LightingMode::Studio,
            hdri: "studio".into(),
            hdri_rotation: 0.0,
            hdri_intensity: 1.0,
            preset: "Three-point (studio)".into(),
            preset_intensity: 1.0,
            matcap: "Clay".into(),
            matcap_use_colors: false,
            follow_model: false,
            exposure: 0.0,
            tonemap: Tonemap::AgX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackgroundStyle {
    Solid,
    Vertical,
    Radial,
    Hdri,
}

impl BackgroundStyle {
    pub fn label(self) -> &'static str {
        match self {
            BackgroundStyle::Solid => "Solid",
            BackgroundStyle::Vertical => "Gradient",
            BackgroundStyle::Radial => "Radial",
            BackgroundStyle::Hdri => "HDRI",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BackgroundSettings {
    pub transparent: bool,
    pub style: BackgroundStyle,
    pub color: [u8; 3],
    pub color2: [u8; 3],
    pub hdri_blur: f32,
    pub ground_shadow: bool,
    pub shadow_strength: f32,
}

impl Default for BackgroundSettings {
    fn default() -> Self {
        Self {
            transparent: false,
            style: BackgroundStyle::Radial,
            color: [0x2a, 0x31, 0x3c],
            color2: [0x10, 0x14, 0x19],
            hdri_blur: 0.4,
            ground_shadow: true,
            shadow_strength: 0.6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionKind {
    Spin,
    Swing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MotionSettings {
    pub kind: MotionKind,
    pub duration: f32,
    pub clockwise: bool,
    pub turns: u32,
    pub swing_angle: f32,
    pub ease: bool,
    pub start_angle: f32,
}

impl Default for MotionSettings {
    fn default() -> Self {
        Self {
            kind: MotionKind::Spin,
            duration: 6.0,
            clockwise: false,
            turns: 1,
            swing_angle: 35.0,
            ease: false,
            start_angle: 0.0,
        }
    }
}

impl MotionSettings {
    /// Model yaw in degrees at loop position `t` in [0, 1).
    pub fn angle_at(&self, t: f32) -> f32 {
        let dir = if self.clockwise { -1.0 } else { 1.0 };
        let offset = match self.kind {
            MotionKind::Spin => {
                let t = if self.ease { ease_in_out(t) } else { t };
                360.0 * self.turns.max(1) as f32 * t
            }
            MotionKind::Swing => self.swing_angle * (t * std::f32::consts::TAU).sin(),
        };
        self.start_angle + dir * offset
    }
}

fn ease_in_out(t: f32) -> f32 {
    0.5 - 0.5 * (t * std::f32::consts::PI).cos()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Gif,
    Mp4,
    Webm,
    ProRes,
    DnxHr,
    Webp,
    Apng,
    PngSequence,
}

impl OutputFormat {
    pub const ALL: [OutputFormat; 8] = [
        OutputFormat::Gif,
        OutputFormat::Mp4,
        OutputFormat::Webm,
        OutputFormat::ProRes,
        OutputFormat::DnxHr,
        OutputFormat::Webp,
        OutputFormat::Apng,
        OutputFormat::PngSequence,
    ];

    pub fn label(self) -> &'static str {
        match self {
            OutputFormat::Gif => "GIF",
            OutputFormat::Mp4 => "MP4 (H.264)",
            OutputFormat::Webm => "WebM (VP9)",
            OutputFormat::ProRes => "MOV (ProRes 4444)",
            OutputFormat::DnxHr => "MOV (DNxHR HQ)",
            OutputFormat::Webp => "Animated WebP",
            OutputFormat::Apng => "Animated PNG",
            OutputFormat::PngSequence => "PNG sequence",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::Gif => "gif",
            OutputFormat::Mp4 => "mp4",
            OutputFormat::Webm => "webm",
            OutputFormat::ProRes | OutputFormat::DnxHr => "mov",
            OutputFormat::Webp => "webp",
            OutputFormat::Apng => "png",
            OutputFormat::PngSequence => "",
        }
    }

    pub fn supports_alpha(self) -> bool {
        !matches!(self, OutputFormat::Mp4 | OutputFormat::DnxHr)
    }

    pub fn needs_ffmpeg(self) -> bool {
        !matches!(self, OutputFormat::Gif | OutputFormat::PngSequence)
    }

    pub fn has_quality(self) -> bool {
        matches!(
            self,
            OutputFormat::Mp4 | OutputFormat::Webm | OutputFormat::Webp
        )
    }

    /// 4:2:0 and 4:2:2 chroma subsampling need even frame dimensions.
    pub fn needs_even_size(self) -> bool {
        matches!(
            self,
            OutputFormat::Mp4 | OutputFormat::Webm | OutputFormat::DnxHr
        )
    }

    pub fn from_extension(ext: &str) -> Option<Self> {
        Some(match ext.to_ascii_lowercase().as_str() {
            "gif" => OutputFormat::Gif,
            "mp4" | "m4v" => OutputFormat::Mp4,
            "webm" => OutputFormat::Webm,
            "mov" => OutputFormat::ProRes,
            "webp" => OutputFormat::Webp,
            "png" | "apng" => OutputFormat::Apng,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputSettings {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub format: OutputFormat,
    pub quality: u8,
    pub supersample: u32,
    pub gif_dither: bool,
}

impl Default for OutputSettings {
    fn default() -> Self {
        Self {
            width: 600,
            height: 600,
            fps: 30,
            format: OutputFormat::Gif,
            quality: 80,
            supersample: 2,
            gif_dither: false,
        }
    }
}

impl OutputSettings {
    pub fn frame_count(&self, duration: f32) -> u32 {
        ((duration * self.fps as f32).round() as u32).max(2)
    }

    pub fn render_size(&self) -> (u32, u32) {
        let (mut w, mut h) = (self.width.max(16), self.height.max(16));
        if self.format.needs_even_size() {
            w += w % 2;
            h += h % 2;
        }
        (w, h)
    }
}

pub const SIZE_PRESETS: &[(&str, u32, u32)] = &[
    ("600 × 600", 600, 600),
    ("1080 × 1080", 1080, 1080),
    ("1920 × 1080", 1920, 1080),
    ("1080 × 1920", 1080, 1920),
    ("1280 × 720", 1280, 720),
    ("3840 × 2160", 3840, 2160),
    ("512 × 512", 512, 512),
];

pub fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn parse_hex_color(s: &str) -> Option<[u8; 3]> {
    let s = s.trim().trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    let v = u32::from_str_radix(s, 16).ok()?;
    Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_loops_seamlessly() {
        let m = MotionSettings::default();
        assert_eq!(m.angle_at(0.0), 0.0);
        assert!((m.angle_at(1.0) - 360.0).abs() < 1e-3);
    }

    #[test]
    fn even_sizes_for_h264() {
        let o = OutputSettings {
            width: 601,
            height: 333,
            format: OutputFormat::Mp4,
            ..Default::default()
        };
        assert_eq!(o.render_size(), (602, 334));
    }
}
