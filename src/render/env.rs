use std::f32::consts::PI;

use anyhow::{Context, Result};
use glam::Vec3;

pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Vec3>,
}

impl HdrImage {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
            .context("decoding HDRI")?
            .into_rgb32f();
        Ok(Self {
            width: img.width(),
            height: img.height(),
            pixels: img
                .pixels()
                .map(|p| Vec3::new(p.0[0], p.0[1], p.0[2]))
                .collect(),
        })
    }

    pub fn sample(&self, dir: Vec3) -> Vec3 {
        let (u, v) = dir_to_equirect(dir);
        let x = ((u * self.width as f32) as i64).rem_euclid(self.width as i64) as u32;
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        self.pixels[(y * self.width + x) as usize]
    }

    /// Projects radiance onto the first 9 real spherical harmonics.
    pub fn sh9(&self) -> [Vec3; 9] {
        let mut sh = [Vec3::ZERO; 9];
        let (w, h) = (self.width as usize, self.height as usize);
        let d_phi = 2.0 * PI / w as f32;
        let d_theta = PI / h as f32;
        for y in 0..h {
            let theta = (y as f32 + 0.5) * d_theta;
            let weight = theta.sin() * d_theta * d_phi;
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let v = (y as f32 + 0.5) / h as f32;
                let d = equirect_to_dir(u, v);
                let c = self.pixels[y * w + x] * weight;
                let basis = [
                    0.282095,
                    0.488603 * d.y,
                    0.488603 * d.z,
                    0.488603 * d.x,
                    1.092548 * d.x * d.y,
                    1.092548 * d.y * d.z,
                    0.315392 * (3.0 * d.z * d.z - 1.0),
                    1.092548 * d.x * d.z,
                    0.546274 * (d.x * d.x - d.y * d.y),
                ];
                for (s, b) in sh.iter_mut().zip(basis) {
                    *s += c * b;
                }
            }
        }
        sh
    }

    /// Box-filtered mip chain as RGBA f16, level 0 first.
    pub fn mip_chain(&self) -> Vec<(u32, u32, Vec<half::f16>)> {
        let mut levels = Vec::new();
        let (mut w, mut h) = (self.width, self.height);
        let mut cur: Vec<Vec3> = self.pixels.clone();
        loop {
            let data = cur
                .iter()
                .flat_map(|c| [c.x, c.y, c.z, 1.0].map(|v| half::f16::from_f32(v.min(60000.0))))
                .collect();
            levels.push((w, h, data));
            if w == 1 && h == 1 {
                break;
            }
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let mut next = vec![Vec3::ZERO; (nw * nh) as usize];
            for y in 0..nh {
                for x in 0..nw {
                    let mut s = Vec3::ZERO;
                    let mut n = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = (x * 2 + dx).min(w - 1);
                            let sy = (y * 2 + dy).min(h - 1);
                            s += cur[(sy * w + sx) as usize];
                            n += 1.0;
                        }
                    }
                    next[(y * nw + x) as usize] = s / n;
                }
            }
            cur = next;
            w = nw;
            h = nh;
        }
        levels
    }
}

pub fn equirect_to_dir(u: f32, v: f32) -> Vec3 {
    let phi = (u - 0.5) * 2.0 * PI;
    let theta = v * PI;
    Vec3::new(
        theta.sin() * phi.sin(),
        theta.cos(),
        -theta.sin() * phi.cos(),
    )
}

pub fn dir_to_equirect(d: Vec3) -> (f32, f32) {
    (
        d.x.atan2(-d.z) / (2.0 * PI) + 0.5,
        d.y.clamp(-1.0, 1.0).acos() / PI,
    )
}

pub const MATCAP_SIZE: u32 = 256;

/// Procedural matcaps, linear RGBA f16. `studio` feeds the reflective ones.
pub fn matcap(name: &str, studio: &HdrImage) -> Vec<half::f16> {
    let s = MATCAP_SIZE;
    let mut out = Vec::with_capacity((s * s * 4) as usize);
    let view = Vec3::Z;
    for y in 0..s {
        for x in 0..s {
            let nx = (x as f32 + 0.5) / s as f32 * 2.0 - 1.0;
            let ny = -((y as f32 + 0.5) / s as f32 * 2.0 - 1.0);
            let r2 = nx * nx + ny * ny;
            let n = if r2 >= 1.0 {
                Vec3::new(nx, ny, 0.0).normalize()
            } else {
                Vec3::new(nx, ny, (1.0 - r2).sqrt())
            };
            let c = shade_matcap(name, n, view, studio);
            out.extend([c.x, c.y, c.z, 1.0].map(half::f16::from_f32));
        }
    }
    out
}

fn shade_matcap(name: &str, n: Vec3, v: Vec3, studio: &HdrImage) -> Vec3 {
    let key = Vec3::new(-0.55, 0.65, 0.55).normalize();
    let fill = Vec3::new(0.7, 0.1, 0.5).normalize();
    let rim = Vec3::new(0.2, 0.4, -0.9).normalize();
    let wrap = |l: Vec3, w: f32| ((n.dot(l) + w) / (1.0 + w)).max(0.0);
    let spec = |l: Vec3, p: f32| {
        let h = (l + v).normalize();
        n.dot(h).max(0.0).powf(p)
    };
    let fres = (1.0 - n.dot(v).max(0.0)).powf(3.0);
    match name {
        "Studio grey" => {
            let base = Vec3::splat(0.55);
            base * (wrap(key, 0.1) * 1.1 + wrap(fill, 0.3) * 0.35 + 0.12)
                + Vec3::splat(spec(key, 60.0) * 0.5 + fres * 0.25)
        }
        "Chrome" => {
            let r = (n * 2.0 * n.dot(v) - v).normalize();
            studio.sample(r) * 0.9 + Vec3::splat(fres * 0.1)
        }
        "Jade" => {
            let base = Vec3::new(0.18, 0.55, 0.38);
            let sss = Vec3::new(0.35, 0.85, 0.55) * wrap(key, 0.8) * 0.5;
            base * wrap(key, 0.4) * 0.8
                + sss
                + Vec3::splat(spec(key, 120.0) * 0.8)
                + Vec3::new(0.5, 0.9, 0.7) * fres * 0.35
        }
        "Toon" => {
            let d = n.dot(key);
            let band = if d > 0.55 {
                1.0
            } else if d > 0.05 {
                0.62
            } else {
                0.32
            };
            let base = Vec3::new(0.85, 0.82, 0.78);
            base * band
                + Vec3::splat(if spec(key, 80.0) > 0.6 { 0.35 } else { 0.0 })
                + Vec3::splat(if fres > 0.55 { 0.2 } else { 0.0 })
        }
        "Normals" => n * 0.5 + Vec3::splat(0.5),
        _ => {
            // Clay
            let base = Vec3::new(0.78, 0.60, 0.50);
            base * (wrap(key, 0.3) * 0.95 + wrap(fill, 0.5) * 0.25 + 0.1)
                + Vec3::new(1.0, 0.9, 0.85) * (spec(key, 18.0) * 0.12)
                + Vec3::new(1.0, 0.85, 0.75) * wrap(rim, 0.0) * fres * 0.6
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equirect_round_trip() {
        for d in [
            Vec3::new(0.3, 0.5, -0.8).normalize(),
            Vec3::new(-0.9, -0.1, 0.4).normalize(),
            Vec3::new(0.0, 0.2, 1.0).normalize(),
        ] {
            let (u, v) = dir_to_equirect(d);
            assert!((equirect_to_dir(u, v) - d).length() < 1e-4);
        }
    }

    #[test]
    fn uniform_env_sh_gives_constant_irradiance() {
        let img = HdrImage {
            width: 64,
            height: 32,
            pixels: vec![Vec3::ONE; 64 * 32],
        };
        let sh = img.sh9();
        // E = pi * L for a uniform environment; L00 term alone carries it.
        let e = sh[0].x * 0.886227;
        assert!((e - PI).abs() < 0.02, "{e}");
        assert!(sh[1].length() < 1e-3);
    }
}
