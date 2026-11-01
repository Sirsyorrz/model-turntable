const PI: f32 = 3.14159265359;

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    spin: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: array<vec4<f32>, 4>,
    light_color: array<vec4<f32>, 4>,
    ambient: vec4<f32>,
    sh: array<vec4<f32>, 9>,
    bg_color: vec4<f32>,
    bg_color2: vec4<f32>,
    clay_color: vec4<f32>,
    // x: mode (0 lights, 1 hdri, 2 matcap), y: env rotation, z: env intensity, w: exposure scale
    p0: vec4<f32>,
    // x: tonemap, y: light count, z: flat shading, w: clay
    p1: vec4<f32>,
    // x: bg style, y: bg blur lod, z: transparent, w: shadow strength
    p2: vec4<f32>,
    // x: ground y, y: shadow half extent, z: env max lod, w: matcap uses model colours
    p3: vec4<f32>,
    // x: clay roughness, y: viewport aspect
    p4: vec4<f32>,
};

struct MaterialU {
    base_color: vec4<f32>,
    // rgb emissive, a occlusion strength
    emissive: vec4<f32>,
    // metallic, roughness, normal scale, alpha cutoff
    params: vec4<f32>,
    // alpha mode (0 opaque, 1 mask, 2 blend), has normal map, unlit
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_sampler: sampler;
@group(0) @binding(3) var matcap_tex: texture_2d<f32>;
@group(0) @binding(4) var shadow_tex: texture_2d<f32>;
@group(0) @binding(5) var clamp_sampler: sampler;

@group(1) @binding(0) var<uniform> mat: MaterialU;
@group(1) @binding(1) var base_tex: texture_2d<f32>;
@group(1) @binding(2) var mr_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var occlusion_tex: texture_2d<f32>;
@group(1) @binding(5) var emissive_tex: texture_2d<f32>;
@group(1) @binding(6) var mat_sampler: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var out: VsOut;
    let world = frame.model * vec4<f32>(v.pos, 1.0);
    out.world_pos = world.xyz;
    out.clip = frame.view_proj * world;
    out.normal = (frame.normal_mat * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

fn rotate_y(d: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}

fn dir_to_equirect(d: vec3<f32>) -> vec2<f32> {
    let u = atan2(d.x, -d.z) / (2.0 * PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

fn sample_env(d: vec3<f32>, lod: f32) -> vec3<f32> {
    let dir = rotate_y(d, -frame.p0.y);
    return textureSampleLevel(env_tex, env_sampler, dir_to_equirect(dir), lod).rgb;
}

fn sh_irradiance(n_world: vec3<f32>) -> vec3<f32> {
    let n = rotate_y(n_world, -frame.p0.y);
    var r = frame.sh[0].rgb * 0.886227;
    r += frame.sh[1].rgb * (1.023328 * n.y);
    r += frame.sh[2].rgb * (1.023328 * n.z);
    r += frame.sh[3].rgb * (1.023328 * n.x);
    r += frame.sh[4].rgb * (0.858086 * n.x * n.y);
    r += frame.sh[5].rgb * (0.858086 * n.y * n.z);
    r += frame.sh[6].rgb * (0.247708 * (3.0 * n.z * n.z - 1.0));
    r += frame.sh[7].rgb * (0.858086 * n.x * n.z);
    r += frame.sh[8].rgb * (0.429043 * (n.x * n.x - n.y * n.y));
    return max(r, vec3<f32>(0.0)) / PI;
}

fn env_brdf_approx(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

fn d_ggx(n_dot_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn v_smith(n_dot_v: f32, n_dot_l: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

fn fresnel(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
}

// Minimal AgX (Benjamin Wrensch), output converted back to linear for the sRGB target.
fn agx(c_in: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var c = m * c_in;
    c = clamp(log2(max(c, vec3<f32>(1e-10))), vec3<f32>(min_ev), vec3<f32>(max_ev));
    c = (c - min_ev) / (max_ev - min_ev);
    let x2 = c * c;
    let x4 = x2 * x2;
    c = 15.5 * x4 * x2 - 40.14 * x4 * c + 31.96 * x4 - 6.868 * x2 * c + 0.4298 * x2 + 0.1191 * c - 0.00232;
    let inv = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    c = inv * c;
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let e = c * frame.p0.w;
    let mode = i32(frame.p1.x);
    if mode == 0 {
        return agx(e);
    } else if mode == 1 {
        return aces(e);
    }
    return clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn perturb_normal(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>, map: vec3<f32>) -> vec3<f32> {
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let len = max(dot(t, t), dot(b, b));
    if len < 1e-20 {
        return n;
    }
    let inv = inverseSqrt(len);
    // glTF normal maps are +Y up in image space, i.e. towards decreasing v.
    return normalize(t * inv * map.x - b * inv * map.y + n * map.z);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let base_sample = textureSample(base_tex, mat_sampler, in.uv);
    let mr_sample = textureSample(mr_tex, mat_sampler, in.uv);
    let normal_sample = textureSample(normal_tex, mat_sampler, in.uv).xyz;
    let occ_sample = textureSample(occlusion_tex, mat_sampler, in.uv).r;
    let emissive_sample = textureSample(emissive_tex, mat_sampler, in.uv).rgb;

    let view_vec = normalize(frame.camera_pos.xyz - in.world_pos);
    var ng = normalize(cross(dpdx(in.world_pos), dpdy(in.world_pos)));
    if dot(ng, view_vec) < 0.0 {
        ng = -ng;
    }
    var n = normalize(in.normal);
    if frame.p1.z > 0.5 || dot(n, n) < 0.5 {
        n = ng;
    } else if dot(n, ng) < 0.0 {
        // Winding in imported meshes is unreliable, so orient by the geometric normal.
        n = -n;
    }
    let pert_normal = perturb_normal(n, in.world_pos, in.uv, (normal_sample * 2.0 - 1.0) * vec3<f32>(mat.params.z, mat.params.z, 1.0));

    let clay = frame.p1.w > 0.5;
    var base = mat.base_color * base_sample * in.color;
    var metallic = mat.params.x * mr_sample.b;
    var roughness = mat.params.y * mr_sample.g;
    var ao = mix(1.0, occ_sample, mat.emissive.a);
    var emissive = mat.emissive.rgb * emissive_sample;
    if mat.flags.y > 0.5 && !clay {
        n = pert_normal;
    }
    if clay {
        base = vec4<f32>(frame.clay_color.rgb, 1.0);
        metallic = 0.0;
        roughness = frame.p4.x;
        ao = 1.0;
        emissive = vec3<f32>(0.0);
    }

    let alpha_mode = i32(mat.flags.x);
    var alpha = base.a;
    if clay || alpha_mode == 0 {
        alpha = 1.0;
    } else if alpha_mode == 1 {
        if alpha < mat.params.w {
            discard;
        }
        alpha = 1.0;
    }

    roughness = clamp(roughness, 0.03, 1.0);
    let n_dot_v = clamp(dot(n, view_vec), 1e-4, 1.0);
    let f0 = mix(vec3<f32>(0.04), base.rgb, metallic);
    let diffuse_color = base.rgb * (1.0 - metallic);

    var color = vec3<f32>(0.0);
    let mode = i32(frame.p0.x);
    if mat.flags.z > 0.5 && !clay {
        color = base.rgb;
    } else if mode == 1 {
        let r = reflect(-view_vec, n);
        let spec = sample_env(r, roughness * frame.p3.z) * env_brdf_approx(f0, roughness, n_dot_v);
        let diffuse = sh_irradiance(n) * diffuse_color;
        color = (diffuse + spec) * ao * frame.p0.z;
    } else if mode == 2 {
        let vn = normalize((frame.view * vec4<f32>(n, 0.0)).xyz);
        let muv = vec2<f32>(vn.x * 0.49 + 0.5, -vn.y * 0.49 + 0.5);
        var mc = textureSampleLevel(matcap_tex, clamp_sampler, muv, 0.0).rgb;
        if frame.p3.w > 0.5 {
            mc *= base.rgb;
        }
        color = mc * ao;
    } else {
        let a = roughness * roughness;
        let count = i32(frame.p1.y);
        for (var i = 0; i < 4; i++) {
            if i >= count {
                break;
            }
            let l = -normalize(frame.light_dir[i].xyz);
            let n_dot_l = dot(n, l);
            if n_dot_l <= 0.0 {
                continue;
            }
            let h = normalize(l + view_vec);
            let n_dot_h = max(dot(n, h), 0.0);
            let v_dot_h = max(dot(view_vec, h), 0.0);
            let f = fresnel(f0, v_dot_h);
            let spec = d_ggx(n_dot_h, a) * v_smith(n_dot_v, n_dot_l, a) * f;
            let kd = (vec3<f32>(1.0) - f) * diffuse_color / PI;
            color += (kd + spec) * frame.light_color[i].rgb * n_dot_l;
        }
        let ambient = frame.ambient.rgb;
        color += ambient * (diffuse_color + env_brdf_approx(f0, roughness, n_dot_v) * 0.5) * ao;
    }
    color += emissive;
    let mapped = tonemap(color);
    return vec4<f32>(mapped * alpha, alpha);
}

struct BgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> BgOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: BgOut;
    out.ndc = uv * 2.0 - 1.0;
    out.clip = vec4<f32>(out.ndc, 0.0, 1.0);
    return out;
}

@fragment
fn fs_background(in: BgOut) -> @location(0) vec4<f32> {
    let style = i32(frame.p2.x);
    if style == 3 {
        let far = frame.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
        let dir = normalize(far.xyz / far.w - frame.camera_pos.xyz);
        let c = sample_env(dir, frame.p2.y * frame.p3.z) * frame.p0.z;
        return vec4<f32>(tonemap(c), 1.0);
    }
    if style == 1 {
        let t = in.ndc.y * 0.5 + 0.5;
        return vec4<f32>(mix(frame.bg_color2.rgb, frame.bg_color.rgb, t), 1.0);
    }
    if style == 2 {
        let p = in.ndc * vec2<f32>(frame.p4.y, 1.0);
        let t = clamp(length(p) / (0.85 * length(vec2<f32>(frame.p4.y, 1.0))), 0.0, 1.0);
        return vec4<f32>(mix(frame.bg_color.rgb, frame.bg_color2.rgb, smoothstep(0.0, 1.0, t)), 1.0);
    }
    return vec4<f32>(frame.bg_color.rgb, 1.0);
}

struct GroundOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_ground(@builtin(vertex_index) i: u32) -> GroundOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[i];
    let ext = frame.p3.y;
    let local = vec4<f32>(c.x * ext, frame.p3.x, c.y * ext, 1.0);
    var out: GroundOut;
    out.clip = frame.view_proj * (frame.spin * local);
    out.uv = c * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_ground(in: GroundOut) -> @location(0) vec4<f32> {
    let s = textureSampleLevel(shadow_tex, clamp_sampler, in.uv, 0.0).r;
    let a = s * frame.p2.w;
    return vec4<f32>(0.0, 0.0, 0.0, a);
}
