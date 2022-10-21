use crate::matrix_util::set_look_at;
use glam::{Mat4, Vec3};

pub struct Camera {
    pub eye: Vec3,
    pub at: Vec3,
    pub up: Vec3,
    pub mat_look_at: Mat4,
}

impl Camera {
    pub fn new(eye: Vec3, at: Vec3, up: Vec3) -> Self {
        Self {
            eye: eye,
            at: at,
            up: up,
            mat_look_at: set_look_at(eye, at, up),
        }
    }

    pub fn cal_look_at(&mut self) -> Mat4{
        self.mat_look_at = set_look_at(self.eye, self.at, self.up);

        self.mat_look_at
    }
}
