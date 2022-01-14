#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub base_color: [f32; 4],
    pub base_color_texture_index: Option<usize>,
}

impl<'a> From<gltf::Material<'a>> for Material {
    fn from(material: gltf::Material) -> Self {
        let pbr = material.pbr_metallic_roughness();
        Self {
            base_color: pbr.base_color_factor(),
            base_color_texture_index: pbr.base_color_texture().map(|i| i.texture().index()),
        }
    }
}
