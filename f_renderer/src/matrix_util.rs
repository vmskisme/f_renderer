use glam::{vec4, Mat4, Vec3, Vec4};

#[inline]
pub fn set_identity() -> Mat4 {
    Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

#[inline]
pub fn set_look_at(eye: Vec3, at: Vec3, up: Vec3) -> Mat4 {
    let z_axis = (at - eye).normalize();
    let x_axis = up.cross(z_axis).normalize();
    let y_axis = z_axis.cross(x_axis);

    Mat4::from_cols(
        Vec4::new(x_axis.x, y_axis.x, z_axis.x, 0.0),
        Vec4::new(x_axis.y, y_axis.y, z_axis.y, 0.0),
        Vec4::new(x_axis.z, y_axis.z, z_axis.z, 0.0),
        Vec4::new(-eye.dot(x_axis), -eye.dot(y_axis), -eye.dot(z_axis), 1.0),
    )
}

#[inline]
pub fn set_perspective(fovy: f32, aspect: f32, zn: f32, zf: f32) -> Mat4 {
    let fax = f32::tan(fovy * 0.5).recip();
    let mut m = Mat4::ZERO;
    m.x_axis[0] = fax / aspect;
    m.y_axis[1] = fax;
    m.z_axis[2] = zf / (zf - zn);
    m.w_axis[2] = -zn * zf / (zf - zn);
    m.z_axis[3] = 1.0;

    m
}

#[inline]
pub fn set_rotate(axis: Vec3, theta: f32) -> Mat4 {
    let (x,y,z) = (axis.x,axis.y,axis.z);
    let q_sin = (theta * 0.5).sin();
    let q_cos = (theta * 0.5).cos();
    let w = q_cos;
    let mut vec = Vec3::new(x, y, z).normalize();
    vec = vec * q_sin;
    let (x, y, z) = (vec.x, vec.y, vec.z);
    Mat4::from_cols(
        vec4(
            1.0 - 2.0 * y * y - 2.0 * z * z,
            2.0 * x * y + 2.0 * w * z,
            2.0 * x * z - 2.0 * w * y,
            0.0,
        ),
        vec4(
            2.0 * x * y - 2.0 * w * z,
            1.0 - 2.0 * x * x - 2.0 * z * z,
            2.0 * y * z + 2.0 * w * x,
            0.0,
        ),
        vec4(
            2.0 * x * z + 2.0 * w * y,
            2.0 * y * z - 2.0 * w * x,
            1.0 - 2.0 * x * x - 2.0 * y * y,
            0.0,
        ),
        vec4(0.0, 0.0, 0.0, 1.0),
    )
}


#[inline]
pub fn set_scale(x:f32,y:f32,z:f32)->Mat4{
    let mut m = set_identity();
    m.x_axis[0] = x;
    m.y_axis[1] = y;
    m.z_axis[2] = z;

    m
}
