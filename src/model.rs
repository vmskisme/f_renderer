use crate::renderer;
use renderer::FrameBuffer;
use glam::{vec2, vec3, UVec3, Vec2, Vec3, Vec4};
use std::fs::File;
use std::io::{prelude::*, IoSlice, IoSliceMut};
use std::path::Path;

pub struct Model {
    verts: Vec<Vec3>,
    faces: Vec<Vec<UVec3>>,
    norms: Vec<Vec3>,
    uv: Vec<Vec2>,
    diffuse_map: FrameBuffer,
    normal_map: FrameBuffer,
    specular_map: FrameBuffer,
}

impl Model {
    pub fn new(path: &str) -> Self {
        let mut model = Self {
            verts: vec![],
            faces: vec![],
            norms: vec![],
            uv: vec![],
            diffuse_map: FrameBuffer::load_file(
                (path.strip_suffix(".obj").unwrap().to_string() + "_diffuse.bmp").as_str(),
            ),
            normal_map: FrameBuffer::load_file(
                (path.strip_suffix(".obj").unwrap().to_string() + "_nm.bmp").as_str(),
            ),
            specular_map: FrameBuffer::load_file(
                (path.strip_suffix(".obj").unwrap().to_string() + "_spec.bmp").as_str(),
            ),
        };

        let mut f = File::open(path).unwrap();
        let mut buffer = String::new();
        f.read_to_string(&mut buffer).unwrap();
        let buffer: Vec<&str> = buffer.split("\n").collect();

        for line in buffer {
            let l_v: Vec<&str> = line.split(" ").collect();
            if l_v.len() < 1 {
                continue;
            }
            match l_v[0] {
                "v" => {
                    model.verts.push(vec3(
                        l_v[1].parse::<f32>().unwrap(),
                        l_v[2].parse::<f32>().unwrap(),
                        l_v[3].parse::<f32>().unwrap(),
                    ));
                }
                "vn" => {
                    model.norms.push(vec3(
                        l_v[2].parse::<f32>().unwrap(),
                        l_v[3].parse::<f32>().unwrap(),
                        l_v[4].parse::<f32>().unwrap(),
                    ));
                }
                "vt" => {
                    model.uv.push(vec2(
                        l_v[2].parse::<f32>().unwrap(),
                        l_v[3].parse::<f32>().unwrap(),
                    ));
                }
                "f" => {
                    let mut v = vec![];
                    for i in 1..4 {
                        let vv: Vec<&str> = l_v[i].split("/").collect();
                        v.push(UVec3::new(
                            vv[0].parse::<u32>().unwrap() - 1,
                            vv[1].parse::<u32>().unwrap() - 1,
                            vv[2].parse::<u32>().unwrap() - 1,
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

impl Model{
    #[inline]
    pub fn diffuse(&self, uv:Vec2) -> Vec4 {
        self.diffuse_map.sample_2d(uv)
    }

    #[inline]
    pub fn faces_len(&self) -> usize{
        self.faces.len()
    }

    #[inline]
    pub fn vert(&self, i_face: usize, nth_vert:usize)->Vec3{
        self.verts[self.faces[i_face][nth_vert][0] as usize]
    }

    #[inline]
    pub fn uv(&self, i_face: usize, nth_vert:usize) -> Vec2{
        self.uv[self.faces[i_face][nth_vert][1] as usize]
    }

    #[inline]
    pub fn normal(&self, i_face: usize, nth_vert:usize)->Vec3{
        self.norms[self.faces[i_face][nth_vert][2] as usize].normalize()
    }
}
