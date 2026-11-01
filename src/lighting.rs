/// Directional light: `dir` is where the light shines toward, Y up, camera on +Z.
pub struct Light {
    pub dir: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
}

pub struct Preset {
    pub name: &'static str,
    pub ambient: f32,
    pub lights: &'static [Light],
    pub headlamp: bool,
    /// Default for "lights spin with model".
    pub follow_model: bool,
}

const fn l(dir: [f32; 3], intensity: f32, color: [f32; 3]) -> Light {
    Light {
        dir,
        intensity,
        color,
    }
}

const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

pub const PRESETS: &[Preset] = &[
    Preset {
        name: "Three-point (studio)",
        ambient: 0.30,
        lights: &[
            l([-1.0, -1.2, -0.8], 4.0, [1.00, 0.98, 0.95]),
            l([1.0, -0.4, -0.6], 1.8, [0.90, 0.95, 1.00]),
            l([0.2, 0.6, 1.0], 2.2, WHITE),
        ],
        headlamp: false,
        follow_model: true,
    },
    Preset {
        name: "Soft even (product shot)",
        ambient: 0.55,
        lights: &[
            l([0.0, -1.0, 0.0], 2.5, WHITE),
            l([-0.7, -0.3, -0.7], 1.5, WHITE),
            l([0.7, -0.3, -0.7], 1.5, WHITE),
            l([0.0, -0.2, 1.0], 1.5, WHITE),
        ],
        headlamp: false,
        follow_model: true,
    },
    Preset {
        name: "Dramatic (single key)",
        ambient: 0.10,
        lights: &[l([-1.0, -1.5, -0.5], 6.0, [1.00, 0.95, 0.85])],
        headlamp: false,
        follow_model: false,
    },
    Preset {
        name: "Top-down (overhead)",
        ambient: 0.25,
        lights: &[
            l([0.0, -1.0, 0.0], 4.5, WHITE),
            l([0.0, -0.4, -1.0], 1.2, WHITE),
        ],
        headlamp: false,
        follow_model: true,
    },
    Preset {
        name: "Rim / backlight",
        ambient: 0.20,
        lights: &[
            l([0.0, 0.3, 1.0], 5.0, WHITE),
            l([-0.5, -0.5, -0.6], 1.2, WHITE),
        ],
        headlamp: false,
        follow_model: true,
    },
    Preset {
        name: "Sunset (warm/cool)",
        ambient: 0.20,
        lights: &[
            l([-1.0, -0.5, -0.3], 4.0, [1.00, 0.70, 0.45]),
            l([1.0, -0.2, 0.5], 2.5, [0.45, 0.60, 1.00]),
        ],
        headlamp: false,
        follow_model: false,
    },
    Preset {
        name: "Headlamp (follows camera)",
        ambient: 0.30,
        lights: &[],
        headlamp: true,
        follow_model: false,
    },
];

pub fn preset(name: &str) -> &'static Preset {
    PRESETS
        .iter()
        .find(|p| p.name == name)
        .unwrap_or(&PRESETS[0])
}

pub struct HdriInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub data: &'static [u8],
}

macro_rules! hdri {
    ($id:literal, $label:literal) => {
        HdriInfo {
            id: $id,
            label: $label,
            data: include_bytes!(concat!("../assets/hdri/", $id, ".hdr")),
        }
    };
}

pub const HDRIS: &[HdriInfo] = &[
    hdri!("studio", "Studio"),
    hdri!("photo_studio", "Photo studio"),
    hdri!("interior", "Interior"),
    hdri!("city", "City"),
    hdri!("forest", "Forest"),
    hdri!("sunset", "Sunset"),
    hdri!("night", "Night"),
];

pub fn hdri(id: &str) -> &'static HdriInfo {
    HDRIS.iter().find(|h| h.id == id).unwrap_or(&HDRIS[0])
}

pub const MATCAPS: &[&str] = &["Clay", "Studio grey", "Chrome", "Jade", "Toon", "Normals"];
