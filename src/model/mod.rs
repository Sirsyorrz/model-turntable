mod fbx;
mod gltf;
mod stl;

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::settings::{Shading, UpAxis};

pub const EXTENSIONS: &[&str] = &["glb", "gltf", "fbx", "obj", "stl"];

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub base_color: [f32; 4],
    pub base_color_tex: Option<usize>,
    pub metallic: f32,
    pub roughness: f32,
    /// glTF layout: roughness in G, metalness in B.
    pub metal_rough_tex: Option<usize>,
    pub normal_tex: Option<usize>,
    pub normal_scale: f32,
    pub occlusion_tex: Option<usize>,
    pub occlusion_strength: f32,
    pub emissive: [f32; 3],
    pub emissive_tex: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub unlit: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0],
            base_color_tex: None,
            metallic: 0.0,
            roughness: 0.5,
            metal_rough_tex: None,
            normal_tex: None,
            normal_scale: 1.0,
            occlusion_tex: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_tex: None,
            alpha_mode: AlphaMode::Opaque,
            unlit: false,
        }
    }
}

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub srgb: bool,
}

impl Texture {
    pub fn from_image(img: image::DynamicImage, srgb: bool) -> Self {
        let rgba = img.into_rgba8();
        Self {
            width: rgba.width(),
            height: rgba.height(),
            rgba: rgba.into_raw(),
            srgb,
        }
    }

    pub fn has_transparency(&self) -> bool {
        self.rgba.as_chunks::<4>().0.iter().any(|p| p[3] < 250)
    }
}

pub struct Model {
    pub path: PathBuf,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub native_up: UpAxis,
    pub default_shading: Shading,
    pub warnings: Vec<String>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Model> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut model = match ext.as_str() {
            "glb" | "gltf" => gltf::load(path)?,
            "fbx" | "obj" => fbx::load(path)?,
            "stl" => stl::load(path)?,
            _ => bail!("Unsupported file type: .{ext}"),
        };
        model.meshes.retain(|m| !m.indices.is_empty());
        if model.meshes.is_empty() {
            bail!("No triangle geometry found in {}", path.display());
        }
        if model.materials.is_empty() {
            model.materials.push(Material::default());
        }
        for mesh in &mut model.meshes {
            if mesh.material >= model.materials.len() {
                mesh.material = 0;
            }
        }
        Ok(model)
    }

    pub fn vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.vertices.len()).sum()
    }

    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.indices.len() / 3).sum()
    }

    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn orientation(&self, up: UpAxis, rotation_deg: [f32; 3]) -> Mat4 {
        let up = if up == UpAxis::Auto {
            self.native_up
        } else {
            up
        };
        let up_fix = if up == UpAxis::Z {
            Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2)
        } else {
            Mat4::IDENTITY
        };
        let [rx, ry, rz] = rotation_deg.map(f32::to_radians);
        let user = Mat4::from_euler(glam::EulerRot::ZYX, rz, ry, rx);
        user * up_fix
    }

    /// Bounds of the model after `orient`, relative to the pivot the turntable spins around.
    pub fn bounds(&self, orient: Mat4) -> Bounds {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for m in &self.meshes {
            for v in &m.vertices {
                let p = orient.transform_point3(Vec3::from(v.pos));
                min = min.min(p);
                max = max.max(p);
            }
        }
        let pivot = (min + max) * 0.5;
        let mut radius_xz: f32 = 0.0;
        let mut radius: f32 = 0.0;
        for m in &self.meshes {
            for v in &m.vertices {
                let p = orient.transform_point3(Vec3::from(v.pos)) - pivot;
                radius_xz = radius_xz.max(p.x * p.x + p.z * p.z);
                radius = radius.max(p.length_squared());
            }
        }
        Bounds {
            pivot,
            radius_xz: radius_xz.sqrt().max(1e-6),
            radius: radius.sqrt().max(1e-6),
            y_min: min.y - pivot.y,
            y_max: max.y - pivot.y,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub pivot: Vec3,
    pub radius_xz: f32,
    pub radius: f32,
    pub y_min: f32,
    pub y_max: f32,
}

pub fn flat_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a).normalize_or(Vec3::Y)
}

/// Fills in missing normals with area-weighted smooth normals.
pub fn compute_smooth_normals(vertices: &mut [Vertex], indices: &[u32]) {
    let mut acc = vec![Vec3::ZERO; vertices.len()];
    for tri in indices.as_chunks::<3>().0 {
        let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| Vec3::from(vertices[i as usize].pos));
        let n = (b - a).cross(c - a);
        for &i in tri {
            acc[i as usize] += n;
        }
    }
    for (v, n) in vertices.iter_mut().zip(acc) {
        v.normal = n.normalize_or(Vec3::Y).into();
    }
}

/// Finds a texture an exporter referenced by a path that may only exist on the
/// author's machine, by looking for the same file name near the model.
pub fn resolve_texture_path(model_dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    for c in candidates.iter().filter(|c| !c.is_empty()) {
        let p = Path::new(c);
        if p.is_absolute() && p.is_file() {
            return Some(p.to_path_buf());
        }
        let rel = model_dir.join(c.replace('\\', "/"));
        if rel.is_file() {
            return Some(rel);
        }
    }
    let names: Vec<&str> = candidates
        .iter()
        .filter_map(|c| c.rsplit(['/', '\\']).next())
        .filter(|n| !n.is_empty())
        .collect();
    let dirs = [
        model_dir.to_path_buf(),
        model_dir.join("textures"),
        model_dir.join("Textures"),
        model_dir.join("tex"),
        model_dir.join("../textures"),
        model_dir.join("../Textures"),
    ];
    for name in &names {
        for d in &dirs {
            let p = d.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
        if let Some(p) = find_case_insensitive(model_dir, name) {
            return Some(p);
        }
    }
    None
}

fn find_case_insensitive(dir: &Path, name: &str) -> Option<PathBuf> {
    let lower = name.to_lowercase();
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().to_lowercase() == lower)
        })
}
