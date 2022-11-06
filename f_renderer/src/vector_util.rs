use glam::Vec3;


#[inline]
pub fn reflect(L: Vec3, N: Vec3) -> Vec3{
    (2.0 * L.dot(N) * N - L).normalize()
}