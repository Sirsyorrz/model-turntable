//! Colours and spacing shared with chaptr and soundbox (their tokens.css).

use std::sync::Arc;

use egui::{
    Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Margin, Shadow, Stroke,
    TextStyle, Vec2, style::WidgetVisuals,
};

pub const BG: Color32 = Color32::from_rgb(0x0e, 0x11, 0x16);
pub const PANEL: Color32 = Color32::from_rgb(0x14, 0x19, 0x20);
pub const PANEL_2: Color32 = Color32::from_rgb(0x10, 0x15, 0x1b);
pub const LINE: Color32 = Color32::from_rgb(0x23, 0x2b, 0x36);
pub const FG: Color32 = Color32::from_rgb(0xd7, 0xde, 0xe8);
pub const FG_DIM: Color32 = Color32::from_rgb(0x8a, 0x94, 0xa2);
pub const FG_FAINT: Color32 = Color32::from_rgb(0x5d, 0x66, 0x73);
pub const ACCENT: Color32 = Color32::from_rgb(0x4e, 0xa1, 0xff);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0x2a, 0x5f, 0x9e);
pub const SEL: Color32 = Color32::from_rgb(0xff, 0xcc, 0x4e);
pub const OK: Color32 = Color32::from_rgb(0x4a, 0xde, 0x80);
pub const BAD: Color32 = Color32::from_rgb(0xf8, 0x71, 0x71);
pub const ROW_SEL: Color32 = Color32::from_rgb(0x1d, 0x29, 0x37);
pub const ROW_HOVER: Color32 = Color32::from_rgb(0x1a, 0x20, 0x29);

pub const RADIUS: u8 = 6;
pub const FS_SM: f32 = 11.0;
pub const FS: f32 = 13.0;
pub const FS_LG: f32 = 15.0;

pub fn semibold() -> FontFamily {
    FontFamily::Name("semibold".into())
}

pub fn apply(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "inter".into(),
        Arc::new(FontData::from_static(include_bytes!(
            "../assets/fonts/Inter-Regular.ttf"
        ))),
    );
    fonts.font_data.insert(
        "inter-semibold".into(),
        Arc::new(FontData::from_static(include_bytes!(
            "../assets/fonts/Inter-SemiBold.ttf"
        ))),
    );
    let proportional = fonts.families.entry(FontFamily::Proportional).or_default();
    proportional.insert(0, "inter".into());
    let mut semi = vec!["inter-semibold".to_owned()];
    semi.extend(proportional.iter().skip(1).cloned());
    fonts.families.insert(semibold(), semi);
    ctx.set_fonts(fonts);

    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Small, FontId::proportional(FS_SM)),
            (TextStyle::Body, FontId::proportional(FS)),
            (TextStyle::Button, FontId::proportional(FS)),
            (TextStyle::Heading, FontId::new(FS_LG, semibold())),
            (TextStyle::Monospace, FontId::monospace(12.0)),
        ]
        .into();

        let s = &mut style.spacing;
        s.item_spacing = Vec2::new(8.0, 6.0);
        s.button_padding = Vec2::new(10.0, 4.0);
        s.interact_size = Vec2::new(24.0, 24.0);
        s.slider_width = 150.0;
        s.slider_rail_height = 4.0;
        s.combo_width = 150.0;
        s.window_margin = Margin::same(12);
        s.menu_margin = Margin::same(4);
        s.icon_width = 14.0;
        s.icon_width_inner = 8.0;

        let v = &mut style.visuals;
        v.dark_mode = true;
        v.override_text_color = None;
        v.panel_fill = BG;
        v.window_fill = PANEL_2;
        v.window_stroke = Stroke::new(1.0, LINE);
        v.window_corner_radius = CornerRadius::same(RADIUS);
        v.menu_corner_radius = CornerRadius::same(RADIUS);
        v.window_shadow = Shadow {
            offset: [0, 8],
            blur: 24,
            spread: 0,
            color: Color32::from_black_alpha(115),
        };
        v.popup_shadow = v.window_shadow;
        v.extreme_bg_color = PANEL_2;
        v.faint_bg_color = PANEL;
        v.code_bg_color = PANEL;
        v.hyperlink_color = ACCENT;
        v.warn_fg_color = SEL;
        v.error_fg_color = BAD;
        v.selection.bg_fill = ACCENT_DIM;
        v.selection.stroke = Stroke::new(1.0, FG);
        v.slider_trailing_fill = true;
        v.handle_shape = egui::style::HandleShape::Circle;
        v.striped = false;
        v.indent_has_left_vline = false;

        let widget = |bg: Color32, border: Color32, fg: Color32| WidgetVisuals {
            bg_fill: bg,
            weak_bg_fill: bg,
            bg_stroke: Stroke::new(1.0, border),
            corner_radius: CornerRadius::same(RADIUS),
            fg_stroke: Stroke::new(1.0, fg),
            expansion: 0.0,
        };
        v.widgets.noninteractive = widget(PANEL, LINE, FG_DIM);
        v.widgets.inactive = widget(PANEL_2, LINE, FG);
        v.widgets.hovered = widget(PANEL_2, ACCENT, FG);
        v.widgets.active = widget(ROW_SEL, ACCENT, FG);
        v.widgets.open = widget(ROW_SEL, ACCENT, FG);
    });
}
