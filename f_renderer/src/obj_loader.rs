use crate::renderer::FrameBuffer;
use glam::{vec2, vec3, UVec3, Vec2, Vec3, Vec4};
use std::fs::File;
use std::io::{prelude::*, IoSlice, IoSliceMut};
use std::path::Path;

pub struct Model {
    verts: Vec<Vec3>,
    faces: Vec<Vec<UVec3>>,
    norms: Vec<Vec3>,
    uv: Vec<Vec2>,
}

impl Model {
    pub fn new(path: &str) -> Self {
        let mut model = Self {
            verts: vec![],
            faces: vec![],
            norms: vec![],
            uv: vec![],
        };
        

        let mut f = File::open(path).unwrap();
        let mut buf = vec![];
        f.read_to_end(&mut buf).unwrap();
        let buffer = String::from_utf8_lossy(&buf).to_string(); // lossy
        let buffer: Vec<&str> = buffer.split("\n").collect();

        for line in buffer {
            let l_v: Vec<&str> = line.split(" ").collect();
            if l_v.len() < 1 {
                continue;
            }
            match l_v[0] {
                "v" => {
                    model.verts.push(vec3(
                        l_v[1].replace("\r", "").parse::<f32>().unwrap(),
                        l_v[2].replace("\r", "").parse::<f32>().unwrap(),
                        l_v[3].replace("\r", "").parse::<f32>().unwrap(),
                    ));
                }
                "vn" => {
                    model.norms.push(vec3(
                        l_v[1].replace("\r", "").parse::<f32>().unwrap(),
                        l_v[2].replace("\r", "").parse::<f32>().unwrap(),
                        l_v[3].replace("\r", "").parse::<f32>().unwrap(),
                    ));
                }
                "vt" => {
                    model.uv.push(vec2(
                        l_v[1].replace("\r", "").parse::<f32>().unwrap(),
                        l_v[2].replace("\r", "").parse::<f32>().unwrap(),
                    ));
                }
                "f" => {
                    let mut v = vec![];
                    for i in 1..4 {
                        let vv: Vec<&str> = l_v[i].split("/").collect();
                        v.push(UVec3::new(
                            vv[0].replace("\r", "").parse::<u32>().unwrap() - 1,
                            vv[1].replace("\r", "").parse::<u32>().unwrap() - 1,
                            vv[2].replace("\r", "").parse::<u32>().unwrap() - 1,
                        ));
                    }
                    model.faces.push(v);
                }
                _ => {}
            }
        }

        println!("v: {}, faces: {}", model.verts.len(), model.faces.len());
        model
    }
}

impl Model {
    #[inline]
    pub fn faces_len(&self) -> usize {
        self.faces.len()
    }

    #[inline]
    pub fn vert(&self, i_face: usize, nth_vert: usize) -> Vec3 {
        self.verts[self.faces[i_face][nth_vert][0] as usize]
    }

    #[inline]
    pub fn uv(&self, i_face: usize, nth_vert: usize) -> Vec2 {
        self.uv[self.faces[i_face][nth_vert][1] as usize]
    }

    #[inline]
    pub fn normal(&self, i_face: usize, nth_vert: usize) -> Vec3 {
        self.norms[self.faces[i_face][nth_vert][2] as usize].normalize()
    }
}
