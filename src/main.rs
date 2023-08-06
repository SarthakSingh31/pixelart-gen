#![feature(get_many_mut)]

mod color;
mod image;

use std::{
    collections::{hash_map::RandomState, VecDeque},
    fs,
    path::PathBuf,
};

use ::image::{Rgb, RgbImage};
use clap::Parser;
use color::Color;
use glam::{DVec2, DVec3, IVec2, UVec2};
use image::LabImage;
use palette::{chromatic_adaptation::AdaptFrom, color_difference::EuclideanDistance};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use tracing::info;

const ALPHA: f64 = 0.7;
const T_FINAL: f64 = 1.0;
const EPSILON_PALETTE: f64 = 1.0;
const EPSILON_CLUSTER: f64 = 0.25;

#[derive(Debug, Parser)]
pub struct Args {
    // Path to the input image
    #[arg(short)]
    input: PathBuf,
    // Path to the output image
    #[arg(short)]
    output: String,
    // Max size of the greater sized side in the output
    #[arg(short)]
    max_side_size: u16,
    // Total color count in the output
    #[arg(short)]
    color_count: u8,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input: LabImage = {
        let bytes = fs::read(args.input)?;
        ::image::load_from_memory(&bytes)?.into()
    };

    println!("{:?}", input[UVec2::new(0, 0)]);

    let out_size = if input.size.x >= input.size.y {
        UVec2 {
            x: args.max_side_size as u32,
            y: ((args.max_side_size as f64 / input.size.x as f64) * (input.size.y as f64)).ceil()
                as u32,
        }
    } else {
        UVec2 {
            x: ((args.max_side_size as f64 / input.size.y as f64) * (input.size.x as f64)).ceil()
                as u32,
            y: args.max_side_size as u32,
        }
    };

    println!("In Size: {:?}, Out Size: {out_size}", input.size);

    let pca = input.pca()?;
    let component = pca.components().axis_iter(ndarray::Axis(0)).next().unwrap();
    let component = component.as_slice().unwrap();

    let delta = DVec3 {
        x: component[0],
        y: component[1],
        z: component[2],
    } * 1.5;
    let mut t = 1.1 * pca.explained_variance().first().unwrap();
    // let mut t = 35.0;
    let mut k = 1;

    let init_color = dbg!(Color::average_from(&input, input.size));
    let mut super_pixels = Vec::with_capacity((out_size.x * out_size.y) as usize);

    for y in (0..out_size.y).map(|y| (y * input.size.y) / out_size.y) {
        for x in (0..out_size.x).map(|x| (x * input.size.x) / out_size.x) {
            super_pixels.push(SuperPixel::new(
                &input,
                UVec2 { x, y },
                init_color,
                out_size,
            ));
        }
    }

    let mut clusters = vec![UVec2 { x: 0, y: 1 }];
    let mut palette = vec![(init_color, 0.5), (init_color, 0.5)];
    palette[1].0.perturb(delta.truncate());

    let dmc_colors = load_dmc_colors();
    let lab_dmc_colors = dmc_colors
        .iter()
        .map(|color| palette::Lab::<palette::white_point::D65, _>::adapt_from(*color))
        .collect::<Vec<_>>();
    let colors: dashmap::DashSet<Rgb<u8>, RandomState> = dashmap::DashSet::default();
    let mut output = RgbImage::new(out_size.x, out_size.y);
    let mut running_average = 0.0;
    let mut prev_changes = VecDeque::with_capacity(100);
    let mut running_variance_avg = 0.0;
    let mut prev_variances = VecDeque::with_capacity(100);
    let mut variance_check_passed_count = 0;

    let mut i = 0;

    while t > T_FINAL {
        let start = std::time::Instant::now();

        sp_refine(&mut super_pixels, input.size, out_size);

        associate(&mut super_pixels, &mut palette, &clusters, k, t);

        let total_change = palette_refine(&mut super_pixels, &mut palette);

        if prev_changes.len() == 100 {
            running_average -= prev_changes.pop_front().unwrap();
        }

        prev_changes.push_back(total_change);
        running_average += total_change;

        let mean = running_average / 100.0;
        let variance = prev_changes
            .iter()
            .map(|change| (mean - change).powi(2))
            .sum::<f64>()
            .sqrt()
            / 100.0;

        if prev_variances.len() == 100 {
            running_variance_avg -= prev_variances.pop_front().unwrap();
        }

        prev_variances.push_back(variance);
        running_variance_avg += variance;

        if ((running_variance_avg / 100.0) - variance).abs() < 0.001 {
            variance_check_passed_count += 1;
            println!("Trigger due to variance");
        } else {
            variance_check_passed_count = 0;
        }

        if total_change < EPSILON_PALETTE || variance_check_passed_count > 100 {
            variance_check_passed_count = 0;
            t *= ALPHA;
            if k < args.color_count as usize {
                expand(
                    &mut clusters,
                    &mut palette,
                    &mut k,
                    args.color_count as usize,
                    delta.truncate(),
                );
            }
        }

        colors.clear();

        let pixels = super_pixels
            .par_iter_mut()
            .map(|sp| sp.palette_color * DVec3::new(1.0, 1.1, 1.1))
            .map(|color| {
                palette::Lab::<palette::white_point::D65, _>::new(color.l(), color.a(), color.b())
            })
            .map(|color| {
                let mut min_distance = f64::MAX;
                let mut min_color = dmc_colors[0];

                for (dmc_color, lab_dmc_color) in dmc_colors.iter().zip(lab_dmc_colors.iter()) {
                    let distance = lab_dmc_color.distance_squared(color);
                    if distance < min_distance {
                        min_color = *dmc_color;
                        min_distance = distance;
                    }
                }

                min_color
            })
            .map(|color: palette::rgb::Srgb<f64>| {
                let color = color.into_format::<u8>();
                colors.insert(Rgb::from([color.red, color.green, color.blue]));
                Rgb::from([color.red, color.green, color.blue])
            });

        pixels
            .zip(output.par_iter_mut().chunks(3))
            .for_each(|(color, mut pixel)| {
                *pixel[0] = color.0[0];
                *pixel[1] = color.0[1];
                *pixel[2] = color.0[2];
            });

        output.save(&args.output)?;

        println!(
            "{i}: Total Change: {total_change:.3}, k: {k}, t: {t:.3}, time_delta: {:?}, color_count: {:?}, variance: {variance:.4}, avg. variance: {:.4} variance count: {variance_check_passed_count}\n",
            start.elapsed(), colors.len(), running_variance_avg / 100.0
        );
        i += 1;
    }

    Ok(())
}

#[derive(Debug)]
pub struct SuperPixel<'s> {
    img: &'s LabImage,
    coord: UVec2,
    palette_color: Color,
    probability: f64,
    pixels: dashmap::DashSet<UVec2>,
    conditional_probability: Vec<f64>,
    sp_color: Color,
    original_coord: UVec2,
    original_color: Color,
    n: f64,
    m: f64,
}

impl<'s> SuperPixel<'s> {
    pub fn new<'i: 's>(img: &'i LabImage, coord: UVec2, color: Color, out_size: UVec2) -> Self {
        SuperPixel {
            img,
            coord,
            palette_color: color,
            probability: 1.0 / (out_size.x * out_size.y) as f64,
            pixels: dashmap::DashSet::default(),
            conditional_probability: vec![0.5, 0.5],
            sp_color: Color::BLACK,
            original_coord: coord,
            original_color: img[coord],
            n: (out_size.x * out_size.y) as f64,
            m: (img.size.x * img.size.y) as f64,
        }
    }

    pub fn cost(&self, coord: UVec2) -> f64 {
        let c_diff = self.img[coord].distance(self.palette_color);
        let spatial_diff = self.coord.as_dvec2().distance(coord.as_dvec2());

        c_diff + 45.0 * (self.n / self.m).powf(0.5) * spatial_diff
    }

    pub fn normalize_probs(
        &mut self,
        palette: &Vec<(Color, f64)>,
        clusters: &Vec<UVec2>,
        k: usize,
    ) {
        let denom: f64 = self.conditional_probability.iter().sum();
        let mut hi = self
            .conditional_probability
            .iter()
            .map(|f| *f)
            .reduce(f64::max)
            .unwrap();

        for (i, probability) in self.conditional_probability.iter_mut().enumerate() {
            if *probability == hi {
                self.palette_color = palette[i].0;
            }

            *probability /= denom;
        }

        hi = -1.0;
        for i in 0..k {
            let cluster = clusters[i];
            let mut prob = 0.0;
            let mut color = Color::BLACK;

            for ci in cluster.to_array() {
                let cur = palette[ci as usize];
                color += cur.0;
                prob += cur.1;
            }

            color /= cluster.to_array().len() as f64;

            if prob > hi {
                hi = prob;
                // self.palette_color = color;
            }
        }
    }

    pub fn update_position(&mut self) {
        if self.pixels.len() == 0 {
            info!("super pixel without pixels failure");
            self.coord = self.original_coord;
        } else {
            self.coord = self.pixels.iter().map(|v| *v).sum::<UVec2>() / self.pixels.len() as u32;
        }
    }

    pub fn update_sp_color(&mut self) {
        if self.pixels.len() == 0 {
            self.sp_color = self.original_color;
        } else {
            self.sp_color = self
                .pixels
                .iter()
                .map(|coord| self.img[*coord])
                .sum::<Color>()
                / self.pixels.len() as f64;
        }
    }
}

fn sp_refine(super_pixels: &mut Vec<SuperPixel>, in_size: UVec2, out_size: UVec2) {
    super_pixels
        .into_par_iter()
        .for_each(|sp| sp.pixels.clear());

    (0..(in_size.x * in_size.y))
        .into_par_iter()
        .for_each(|idx| {
            let coord = UVec2 {
                x: idx % in_size.x,
                y: idx / in_size.x,
            };
            let sp_coord = (coord * out_size) / in_size;
            const D_COORDS: [IVec2; 9] = [
                IVec2::new(-1, -1),
                IVec2::new(-1, 0),
                IVec2::new(-1, 1),
                IVec2::new(0, -1),
                IVec2::new(0, 0),
                IVec2::new(0, 1),
                IVec2::new(1, -1),
                IVec2::new(1, 0),
                IVec2::new(1, 1),
            ];

            let mut best_cost = f64::MAX;
            let mut best_coord = UVec2::ZERO;
            for d_coord in D_COORDS {
                let n_coord = sp_coord.as_ivec2() + d_coord;
                if n_coord.x >= 0
                    && n_coord.y >= 0
                    && n_coord.x < out_size.x as i32
                    && n_coord.y < out_size.y as i32
                {
                    let n_coord = n_coord.as_uvec2();
                    let new_cost =
                        super_pixels[(n_coord.x + n_coord.y * out_size.x) as usize].cost(coord);
                    if new_cost < best_cost {
                        best_cost = new_cost;
                        best_coord = n_coord;
                    }
                }
            }

            super_pixels[(best_coord.x + best_coord.y * out_size.x) as usize]
                .pixels
                .insert(coord);
        });

    super_pixels.into_par_iter().for_each(|sp| {
        sp.update_position();
        sp.update_sp_color();
    });

    // Laplacian smoothing
    let mut new_coords = (0..(out_size.x * out_size.y))
        .map(|_| UVec2::ZERO)
        .collect::<Vec<_>>();

    for j in 0..out_size.y {
        for i in 0..out_size.x {
            const D_COORDS: [IVec2; 4] = [
                IVec2::new(0, 1),
                IVec2::new(0, -1),
                IVec2::new(-1, 0),
                IVec2::new(1, 0),
            ];
            let sp = &super_pixels[(i + j * out_size.x) as usize];
            let mut n = 0;
            let mut new = UVec2::ZERO;

            for coord in D_COORDS {
                let n_coord = IVec2::new(i as i32, j as i32) + coord;
                if n_coord.x >= 0
                    && n_coord.y >= 0
                    && n_coord.x < out_size.x as i32
                    && n_coord.y < out_size.y as i32
                {
                    let n_coord = n_coord.as_uvec2();
                    n += 1;

                    new += super_pixels[(n_coord.x + n_coord.y * out_size.x) as usize].coord;
                }
            }

            let mut new = new.as_dvec2();
            new /= n as f64;

            new_coords[(i + j * out_size.x) as usize] =
                (0.4 * new + 0.6 * sp.coord.as_dvec2()).as_uvec2();
        }
    }

    // Bilateral Filter Approximation
    let mut new_colors = (0..(out_size.x * out_size.y))
        .map(|_| Color::BLACK)
        .collect::<Vec<_>>();
    for j in 0..out_size.y {
        for i in 0..out_size.x {
            const D_COORDS: [IVec2; 9] = [
                IVec2::new(-1, -1),
                IVec2::new(-1, 0),
                IVec2::new(-1, 1),
                IVec2::new(0, -1),
                IVec2::new(0, 0),
                IVec2::new(0, 1),
                IVec2::new(1, -1),
                IVec2::new(1, 0),
                IVec2::new(1, 1),
            ];
            let sp = &super_pixels[(i + j * out_size.x) as usize];
            let mut n = 0.0;
            let mut avg_color = Color::BLACK;

            for coord in D_COORDS {
                let n_coord = IVec2::new(i as i32, j as i32) + coord;
                if n_coord.x >= 0
                    && n_coord.y >= 0
                    && n_coord.x < out_size.x as i32
                    && n_coord.y < out_size.y as i32
                {
                    let n_coord = n_coord.as_uvec2();

                    let next = super_pixels[(n_coord.x + n_coord.y * out_size.x) as usize].sp_color;
                    let weight =
                        std::f64::consts::E.powf(-1.0 * (sp.sp_color.l() - next.l()).abs());

                    avg_color += next * weight;

                    n += weight;
                }
            }

            avg_color /= n;

            new_colors[(i + j * out_size.x) as usize] = sp.sp_color * 0.5 + avg_color * 0.5;
        }
    }

    for (i, sp) in super_pixels.iter_mut().enumerate() {
        sp.coord = new_coords[i];
        sp.sp_color = new_colors[i];
    }
}

fn associate(
    super_pixels: &mut Vec<SuperPixel>,
    palettes: &mut Vec<(Color, f64)>,
    clusters: &Vec<UVec2>,
    k: usize,
    t: f64,
) {
    super_pixels.into_par_iter().for_each(|sp| {
        sp.conditional_probability.resize(palettes.len(), 0.0);
        for (i, palette) in palettes.iter().enumerate() {
            sp.conditional_probability[i] = palette.0.condit_prob(palette.1, sp, t);
        }
        sp.normalize_probs(palettes, clusters, k);
    });

    palettes
        .into_par_iter()
        .enumerate()
        .for_each(|(i, palette)| {
            palette.1 = 0.0;

            for sp in super_pixels.iter() {
                palette.1 += sp.conditional_probability[i] * sp.probability;
            }
        });
}

fn palette_refine(super_pixels: &mut Vec<SuperPixel>, palettes: &mut Vec<(Color, f64)>) -> f64 {
    palettes
        .into_par_iter()
        .enumerate()
        .map(|(i, palette)| {
            let mut new_color = Color::BLACK;

            for sp in super_pixels.iter() {
                new_color +=
                    (sp.sp_color * sp.conditional_probability[i] * sp.probability) / palette.1;
            }

            let distance = palette.0.distance(new_color);
            palette.0 = new_color;
            distance
        })
        .sum()
}

fn expand(
    clusters: &mut Vec<UVec2>,
    palettes: &mut Vec<(Color, f64)>,
    k: &mut usize,
    k_max: usize,
    delta: DVec2,
) {
    for i in 0..(*k).min(k_max) {
        let [c1, c2] = palettes
            .get_many_mut([clusters[i].x as usize, clusters[i].y as usize])
            .unwrap();

        if c1.0.distance(c2.0) > EPSILON_CLUSTER {
            *k += 1;

            c1.1 /= 2.0;
            c2.1 /= 2.0;

            let c1 = *c1;
            let c2 = *c2;

            palettes.push(c1);
            palettes.push(c2);

            clusters.push(UVec2::new(clusters[i][1], (palettes.len() - 1) as u32));
            clusters[i] = UVec2::new(clusters[i][0], (palettes.len() - 2) as u32);

            assert!(
                (palettes[clusters[i].x as usize].1 - palettes[clusters[i].y as usize].1).abs()
                    < EPSILON_CLUSTER
            );
            assert!(
                (palettes[clusters.last().unwrap().x as usize].1
                    - palettes[clusters.last().unwrap().y as usize].1)
                    .abs()
                    < EPSILON_CLUSTER
            );
        }
    }

    if *k >= k_max {
        let mut new_palette = Vec::default();
        let mut new_clusters = Vec::default();

        for i in 0..(*k) {
            let c1 = palettes[clusters[i].x as usize];
            let c2 = palettes[clusters[i].y as usize];
            let new_color = (c1.0 + c2.0) / 2.0;

            new_palette.push((new_color, c1.1 + c2.1));
            new_clusters.push(UVec2 { x: i as u32, y: 0 });
        }

        *palettes = new_palette;
        *clusters = new_clusters;
    } else {
        for i in 0..(*k) {
            let c = &mut palettes[clusters[i].y as usize];
            c.0.perturb(delta);
        }
    }
}

fn load_dmc_colors() -> Vec<palette::rgb::Srgb<f64>> {
    #[derive(serde::Deserialize)]
    struct DmcColor {
        red: u8,
        green: u8,
        blue: u8,
    }

    let colors: Vec<DmcColor> = serde_json::from_str(include_str!("../dmc_colors.json")).unwrap();

    colors
        .into_iter()
        .map(|DmcColor { red, green, blue }| palette::rgb::Rgb::new(red, green, blue).into_format())
        .collect()
}
