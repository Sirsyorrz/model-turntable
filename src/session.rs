use std::{collections::HashMap, path::PathBuf, sync::Arc};

use glam::Mat4;

use crate::{
    export::Job,
    lighting,
    model::Model,
    render::{
        Environment, FrameResources, FrameUniforms, GpuModel, GroundShadow, Matcap, Renderer,
        env::{self, HdrImage},
        scene::{self, Framing},
    },
    settings::{Settings, Shading, UpAxis},
};

pub struct LoadedModel {
    pub model: Model,
    pub gpu: Arc<GpuModel>,
}

/// GPU-side state shared by the GUI preview, exports and the CLI.
pub struct Session {
    pub renderer: Arc<Renderer>,
    envs: HashMap<String, Arc<Environment>>,
    matcaps: HashMap<String, Arc<Matcap>>,
    studio_hdr: Option<HdrImage>,
    pub model: Option<LoadedModel>,
    framing: Option<((UpAxis, [f32; 3]), Framing)>,
    shadow: Option<(Mat4, Arc<GroundShadow>)>,
    empty_shadow: Arc<GroundShadow>,
}

pub struct Prepared {
    pub gpu: Arc<GpuModel>,
    pub framing: Framing,
    pub resources: FrameResources,
    pub flat: bool,
}

impl Prepared {
    pub fn uniforms(
        &self,
        s: &Settings,
        aspect: f32,
        angle_deg: f32,
        transparent: bool,
    ) -> FrameUniforms {
        scene::build(scene::SceneInputs {
            settings: s,
            framing: &self.framing,
            env: &self.resources.env,
            flat: self.flat,
            aspect,
            angle_deg,
            transparent,
        })
    }
}

impl Session {
    pub fn new(renderer: Arc<Renderer>) -> Session {
        let empty_shadow = Arc::new(renderer.empty_shadow());
        Session {
            renderer,
            envs: HashMap::new(),
            matcaps: HashMap::new(),
            studio_hdr: None,
            model: None,
            framing: None,
            shadow: None,
            empty_shadow,
        }
    }

    pub fn set_model(&mut self, mut model: Model) {
        let gpu = Arc::new(self.renderer.upload_model(&model));
        model.textures.clear();
        model.textures.shrink_to_fit();
        self.model = Some(LoadedModel { model, gpu });
        self.framing = None;
        self.shadow = None;
    }

    pub fn environment(&mut self, id: &str) -> Arc<Environment> {
        let info = lighting::hdri(id);
        if let Some(e) = self.envs.get(info.id) {
            return e.clone();
        }
        let img = HdrImage::decode(info.data).expect("bundled HDRI decodes");
        let env = Arc::new(self.renderer.create_environment(&img));
        self.envs.insert(info.id.to_owned(), env.clone());
        env
    }

    fn studio_hdr(&mut self) -> &HdrImage {
        self.studio_hdr.get_or_insert_with(|| {
            HdrImage::decode(lighting::hdri("studio").data).expect("bundled HDRI decodes")
        })
    }

    pub fn matcap_pixels(&mut self, name: &str) -> Vec<half::f16> {
        env::matcap(name, self.studio_hdr())
    }

    pub fn matcap(&mut self, name: &str) -> Arc<Matcap> {
        if let Some(m) = self.matcaps.get(name) {
            return m.clone();
        }
        let pixels = self.matcap_pixels(name);
        let m = Arc::new(self.renderer.create_matcap(&pixels));
        self.matcaps.insert(name.to_owned(), m.clone());
        m
    }

    pub fn prepare(&mut self, s: &Settings) -> Option<Prepared> {
        let loaded = self.model.as_ref()?;
        let key = (s.model.up, s.model.rotation);
        let framing = match &self.framing {
            Some((k, f)) if *k == key => *f,
            _ => {
                let orient = loaded.model.orientation(s.model.up, s.model.rotation);
                let f = Framing::new(orient, &loaded.model.bounds(orient));
                self.framing = Some((key, f));
                f
            }
        };
        let flat = match s.model.shading {
            Shading::Auto => loaded.model.default_shading == Shading::Flat,
            Shading::Flat => true,
            Shading::Smooth => false,
        };
        let gpu = loaded.gpu.clone();

        let shadow = if s.background.ground_shadow {
            match &self.shadow {
                Some((local, sh)) if *local == framing.local => sh.clone(),
                _ => {
                    let sh = Arc::new(self.renderer.bake_shadow(&gpu, framing.local, &framing));
                    self.shadow = Some((framing.local, sh.clone()));
                    sh
                }
            }
        } else {
            self.empty_shadow.clone()
        };

        let env = self.environment(&s.lighting.hdri);
        let matcap = self.matcap(&s.lighting.matcap);
        Some(Prepared {
            gpu,
            framing,
            resources: FrameResources {
                env,
                matcap,
                shadow,
            },
            flat,
        })
    }

    pub fn job(&mut self, s: &Settings, out_path: PathBuf, ffmpeg: Option<PathBuf>) -> Option<Job> {
        let p = self.prepare(s)?;
        Some(Job {
            renderer: self.renderer.clone(),
            model: p.gpu,
            resources: p.resources,
            framing: p.framing,
            flat: p.flat,
            settings: s.clone(),
            out_path,
            ffmpeg,
        })
    }
}
