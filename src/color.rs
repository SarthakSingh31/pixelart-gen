use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul},
};

use glam::{DVec2, DVec3, UVec2};

use crate::{image::LabImage, SuperPixel};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Color(DVec3);

impl Color {
    pub const BLACK: Color = Color(DVec3::ZERO);

    pub fn new(l: f64, a: f64, b: f64) -> Self {
        Color(DVec3 { x: l, y: a, z: b })
    }

    pub fn to_array(&self) -> [f64; 3] {
        [self.0.x, self.0.y, self.0.z]
    }

    pub fn distance(&self, rhs: Color) -> f64 {
        self.0.distance(rhs.0)
    }

    pub fn average_from(img: &LabImage, in_size: UVec2) -> Color {
        img.pixels.iter().map(|color| *color).sum::<Color>() / (in_size.x * in_size.y) as f64
    }

    pub fn condit_prob(&self, probability: f64, sp: &SuperPixel, t: f64) -> f64 {
        probability * std::f64::consts::E.powf(-1.0 * sp.sp_color.distance(*self) / t)
    }

    pub fn perturb(&mut self, delta: DVec2) {
        self.0.x += delta.x;
        self.0.y += delta.y;
        self.0.z += delta.y;
    }

    pub fn l(&self) -> f64 {
        self.0.x
    }

    pub fn a(&self) -> f64 {
        self.0.y
    }

    pub fn b(&self) -> f64 {
        self.0.z
    }
}

impl Add for Color {
    type Output = Color;

    fn add(self, rhs: Self) -> Self::Output {
        Color(self.0 + rhs.0)
    }
}

impl AddAssign<Color> for Color {
    fn add_assign(&mut self, rhs: Color) {
        self.0 += rhs.0;
    }
}

impl Sum for Color {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut color = Color::BLACK;
        for c in iter {
            color += c;
        }
        color
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, rhs: f64) -> Self::Output {
        Color(self.0 * rhs)
    }
}

impl Mul<DVec3> for Color {
    type Output = Color;

    fn mul(self, rhs: DVec3) -> Self::Output {
        Color(self.0 * rhs)
    }
}

impl Div<f64> for Color {
    type Output = Color;

    fn div(self, rhs: f64) -> Self::Output {
        Color(self.0 / rhs)
    }
}

impl DivAssign<f64> for Color {
    fn div_assign(&mut self, rhs: f64) {
        self.0 /= rhs;
    }
}
