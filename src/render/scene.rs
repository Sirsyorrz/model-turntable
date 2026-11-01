use bytemuck::{Pod, Zeroable};
use glam::{
    Mat4, Vec3,
    camera::rh::{proj::directx, view::look_at_mat4},
};

use super::Environment;
use crate::{
    lighting,
    model::Bounds,
    settings::{BackgroundStyle, LightingMode, Settings, Tonemap, srgb_to_linear},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub view: Mat4,
    pub model: Mat4,
    pub normal_mat: Mat4,
    pub spin: Mat4,
    pub camera_pos: [f32; 4],
    pub light_dir: [[f32; 4]; 4],
    pub light_color: [[f32; 4]; 4],
    pub ambient: [f32; 4],
    pub sh: [[f32; 4]; 9],
    pub bg_color: [f32; 4],
    pub bg_color2: [f32; 4],
    pub clay_color: [f32; 4],
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    pub p2: [f32; 4],
    pub p3: [f32; 4],
    pub p4: [f32; 4],
}

/// Everything about the model's placement that only changes with orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Framing {
    /// Raw model space -> local space (oriented, pivot at origin).
    pub local: Mat4,
    pub radius_xz: f32,
    pub radius: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub shadow_extent: f32,
}

impl Framing {
    pub fn new(orient: Mat4, b: &Bounds) -> Framing {
        Framing {
            local: Mat4::from_translation(-b.pivot) * orient,
            radius_xz: b.radius_xz,
            radius: b.radius,
            y_min: b.y_min,
            y_max: b.y_max,
            shadow_extent: b.radius_xz * 1.35 + (b.y_max - b.y_min) * 0.15,
        }
    }

    /// Distance at which the volume swept by a full turn fits the frame.
    fn fit_distance(&self, elevation: f32, fov_y: f32, aspect: f32) -> f32 {
        let tan_y = (fov_y * 0.5).tan();
        let tan_x = tan_y * aspect;
        let (se, ce) = elevation.sin_cos();
        let up = Vec3::new(0.0, ce, -se);
        let back = Vec3::new(0.0, se, ce);
        let mut d: f32 = 0.0;
        const STEPS: usize = 32;
        for i in 0..STEPS {
            let a = i as f32 / STEPS as f32 * std::f32::consts::TAU;
            for y in [self.y_min, self.y_max] {
                let p = Vec3::new(a.cos() * self.radius_xz, y, a.sin() * self.radius_xz);
                let z = p.dot(back);
                d = d
                    .max(z + p.x.abs() / tan_x)
                    .max(z + p.dot(up).abs() / tan_y);
            }
        }
        d * 1.06
    }
}

pub struct SceneInputs<'a> {
    pub settings: &'a Settings,
    pub framing: &'a Framing,
    pub env: &'a Environment,
    pub flat: bool,
    pub aspect: f32,
    pub angle_deg: f32,
    pub transparent: bool,
}

pub fn build(inp: SceneInputs) -> FrameUniforms {
    let s = inp.settings;
    let f = inp.framing;
    let spin = Mat4::from_rotation_y(inp.angle_deg.to_radians());
    let model = spin * f.local;

    let elevation = s.camera.elevation.clamp(-89.0, 89.0).to_radians();
    let fov = s.camera.fov.clamp(5.0, 120.0).to_radians();
    let dist = f.fit_distance(elevation, fov, inp.aspect) / s.camera.zoom.max(0.05);
    let height = f.y_max - f.y_min;
    let target = Vec3::new(0.0, -s.camera.offset_y * height, 0.0);
    let (se, ce) = elevation.sin_cos();
    let eye = target + Vec3::new(0.0, se, ce) * dist;
    let view = look_at_mat4(eye, target, Vec3::Y);
    let near = (dist - f.radius * 1.5).max(dist * 0.01);
    let far = dist + f.radius * 3.0 + f.shadow_extent * 2.0;
    let proj = directx::perspective(fov, inp.aspect, near, far);
    let view_proj = proj * view;

    let follow = s.lighting.follow_model;
    let light_spin = if follow { spin } else { Mat4::IDENTITY };
    let mut light_dir = [[0.0; 4]; 4];
    let mut light_color = [[0.0; 4]; 4];
    let mut light_count = 0;
    let preset = lighting::preset(&s.lighting.preset);
    let k = s.lighting.preset_intensity;
    for (i, l) in preset.lights.iter().take(4).enumerate() {
        let d = light_spin.transform_vector3(Vec3::from(l.dir)).normalize();
        light_dir[i] = [d.x, d.y, d.z, 0.0];
        light_color[i] = [
            l.color[0] * l.intensity * k,
            l.color[1] * l.intensity * k,
            l.color[2] * l.intensity * k,
            0.0,
        ];
        light_count = i + 1;
    }
    if preset.headlamp && light_count < 4 {
        let d = (target - eye).normalize();
        light_dir[light_count] = [d.x, d.y, d.z, 0.0];
        light_color[light_count] = [3.5 * k, 3.5 * k, 3.5 * k, 0.0];
        light_count += 1;
    }
    let ambient = preset.ambient * k;

    let mode = match s.lighting.mode {
        LightingMode::Lights => 0.0,
        LightingMode::Studio => 1.0,
        LightingMode::Matcap => 2.0,
    };
    let mut env_rot = s.lighting.hdri_rotation.to_radians();
    if follow {
        env_rot += inp.angle_deg.to_radians();
    }

    let lin = |c: [u8; 3]| {
        [
            srgb_to_linear(c[0]),
            srgb_to_linear(c[1]),
            srgb_to_linear(c[2]),
            1.0,
        ]
    };
    let bg_style = match s.background.style {
        BackgroundStyle::Solid => 0.0,
        BackgroundStyle::Vertical => 1.0,
        BackgroundStyle::Radial => 2.0,
        BackgroundStyle::Hdri => 3.0,
    };
    let shadow_strength = if s.camera.elevation < 0.0 {
        0.0
    } else {
        s.background.shadow_strength
    };

    let mut sh = [[0.0; 4]; 9];
    for (o, c) in sh.iter_mut().zip(inp.env.sh) {
        *o = [c.x, c.y, c.z, 0.0];
    }

    FrameUniforms {
        view_proj,
        inv_view_proj: view_proj.inverse(),
        view,
        model,
        normal_mat: Mat4::from_mat3(glam::Mat3::from_mat4(model).inverse().transpose()),
        spin,
        camera_pos: [eye.x, eye.y, eye.z, 1.0],
        light_dir,
        light_color,
        ambient: [ambient, ambient, ambient, 0.0],
        sh,
        bg_color: lin(s.background.color),
        bg_color2: lin(s.background.color2),
        clay_color: lin(s.model.clay_color),
        p0: [
            mode,
            env_rot,
            s.lighting.hdri_intensity,
            2f32.powf(s.lighting.exposure),
        ],
        p1: [
            match s.lighting.tonemap {
                Tonemap::AgX => 0.0,
                Tonemap::Aces => 1.0,
                Tonemap::Standard => 2.0,
            },
            light_count as f32,
            inp.flat as u32 as f32,
            s.model.clay as u32 as f32,
        ],
        p2: [
            bg_style,
            s.background.hdri_blur,
            inp.transparent as u32 as f32,
            shadow_strength,
        ],
        p3: [
            f.y_min,
            f.shadow_extent,
            inp.env.max_lod,
            s.lighting.matcap_use_colors as u32 as f32,
        ],
        p4: [s.model.clay_roughness, inp.aspect, 0.0, 0.0],
    }
}
