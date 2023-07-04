use std::ops::{Index, IndexMut};

use glam::UVec2;
use palette::FromColor;

use crate::color::Color;

#[derive(Debug)]
pub struct LabImage {
    // x => l, y => a, z => b
    pub pixels: Vec<Color>,
    pub size: UVec2,
}

impl LabImage {
    fn coord_to_idx(&self, coord: UVec2) -> usize {
        (coord.x + self.size.x * coord.y) as usize
    }

    pub fn pca(
        &self,
    ) -> anyhow::Result<petal_decomposition::RandomizedPca<f64, rand_pcg::Mcg128Xsl64>> {
        let pixels = self
            .pixels
            .iter()
            .map(|pixel| pixel.to_array())
            .collect::<Vec<_>>();
        let arr = ndarray::arr2(&pixels);
        let mut pca = petal_decomposition::RandomizedPcaBuilder::new(1)
            .centering(true)
            .build();
        pca.fit(&arr)?;

        Ok(pca)
    }
}

impl From<image::DynamicImage> for LabImage {
    fn from(img: image::DynamicImage) -> Self {
        let img = img.to_rgb8();
        let size = UVec2 {
            x: img.width(),
            y: img.height(),
        };
        let pixels = img
            .pixels()
            .map(|pixel| {
                let color: palette::rgb::Srgb<f64> =
                    palette::rgb::Srgb::new(pixel.0[0], pixel.0[1], pixel.0[2]).into_format();
                palette::Lab::from_color(color)
            })
            .map(|pixel| Color::new(pixel.l, pixel.a, pixel.b))
            .collect::<Vec<_>>();

        assert_eq!(pixels.len(), (size.x * size.y) as usize);

        LabImage { pixels, size }
    }
}

impl Index<UVec2> for LabImage {
    type Output = Color;

    fn index(&self, index: UVec2) -> &Self::Output {
        &self.pixels[self.coord_to_idx(index)]
    }
}

impl IndexMut<UVec2> for LabImage {
    fn index_mut(&mut self, index: UVec2) -> &mut Self::Output {
        let idx = self.coord_to_idx(index);
        &mut self.pixels[idx]
    }
}
