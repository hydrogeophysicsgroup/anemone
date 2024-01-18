#include <torch/extension.h>

int working(int x){
    return x;
}

PYBIND11_MODULE(cphysics, m) {
  m.def("working", &working, "test if working");
}

// constexpr double mu0 = M_PI * 4e-7;
// constexpr double epsilon0 = 8.854187817e-12;

// tc::Tensor calc_z(const tc::Tensor& sigma, const tc::Tensor& omega, const tc::Tensor& mu, const tc::Tensor& epsilon) {
//     tc::Tensor z = tc::complex<double>(0, mu * omega);
//     return z;
// }

// tc::Tensor calc_y(const tc::Tensor& sigma, const tc::Tensor& omega, const tc::Tensor& mu, const tc::Tensor& epsilon) {
//     tc::Tensor y = tc::complex<double>(sigma, epsilon * omega);
//     return y;
// }

// tc::Tensor calc_wavenumber(const tc::Tensor& sigma, const tc::Tensor& omega, double mu = mu0, double epsilon = epsilon0, bool sqrt = true) {
//     tc::Tensor k = mu * omega * (epsilon * omega - 1j * sigma);
//     if (sqrt) k = tc::sqrt(k);
//     return k;
// }

// tc::Tensor freespace_u0(const tc::Tensor& omega, const tc::Tensor& wave_length, bool sqrt = true) {
//     tc::Tensor k2 = mu0 * epsilon0 * omega.pow(2);
//     tc::Tensor u02 = wave_length.pow(2) - k2;
//     if (sqrt) u02 = tc::sqrt(u02);
//     return u02;
// }

// tc::Tensor refl_transverse_electric(const tc::Tensor& sigma, const tc::Tensor& thickness, const tc::Tensor& omega, const tc::Tensor& wave_length, const tc::Tensor& mu, const tc::Tensor& epsilon) {
//     tc::Tensor mu_tensor = tc::ones_like(sigma) * mu;
//     tc::Tensor epsilon_tensor = tc::ones_like(sigma) * epsilon;

//     int n_layers = sigma.numel();

//     tc::Tensor lambda2 = wave_length.pow(2);

//     tc::Tensor adm_boundary = tc::zeros({ sigma.numel(), omega.numel(), wave_length.numel() }, tc::kComplexDouble);

//     tc::Tensor adm_freespace = tc::sqrt(lambda2 - omega.pow(2) * mu0 * epsilon0) / tc::complex<double>(0, omega * mu0);

//     tc::Tensor k2 = calc_wavenumber(sigma, omega, mu_tensor, epsilon_tensor, false);
//     tc::Tensor zn = calc_z(sigma, omega, mu_tensor, epsilon_tensor);

//     tc::Tensor adm_intrinsic = tc::sqrt(lambda2 - k2) / zn;
//     tc::Tensor tanh_un_thick = tc::tanh(adm_intrinsic.slice(0, 0, n_layers - 1) * thickness);
//     adm_intrinsic /= zn;

//     adm_boundary[n_layers-1] = adm_intrinsic[n_layers-1] * 1.0;

//     for (int i = n_layers - 2; i >= 0; --i) {
//         adm_boundary[i] = adm_intrinsic[i] * (adm_boundary[i + 1] + adm_intrinsic[i] * tanh_un_thick[i]) /
//                           (adm_intrinsic[i] + adm_boundary[i + 1] * tanh_un_thick[i]);
//     }

//     tc::Tensor rte = (adm_freespace - adm_boundary[0]) / (adm_freespace + adm_boundary[0]);
//     return rte;
// }

// tc::Tensor refl_transverse_magnetic(const tc::Tensor& sigma, const tc::Tensor& thickness, const tc::Tensor& omega, const tc::Tensor& wave_length, double mu = mu0, double epsilon = epsilon0) {
//     tc::Tensor mu_tensor = tc::ones_like(sigma) * mu;
//     tc::Tensor epsilon_tensor = tc::ones_like(sigma) * epsilon;

//     int n_layers = sigma.numel();

//     tc::Tensor lambda2 = wave_length.pow(2);

//     tc::Tensor imp_boundary  = tc::zeros({ sigma.numel(), omega.numel(), wave_length.numel() }, tc::kComplexDouble);
//     tc::Tensor imp_freespace = tc::sqrt(lambda2 - omega.pow(2) * mu0 * epsilon0) / tc::complex<double>(0, omega * mu0);

//     tc::Tensor k2 = calc_wavenumber(sigma, omega, mu, epsilon, false);
//     tc::Tensor yn = calc_y(sigma, omega, mu, epsilon);

//     tc::Tensor imp_intrinsic = tc::sqrt(lambda2 - k2);
//     tc::Tensor tanh_un_thick = tc::tanh(imp_intrinsic.slice(0, 0, n_layers - 1) * thickness);
//     imp_intrinsic = imp_intrinsic / yn;

//     imp_boundary[n_layers-1] = imp_intrinsic[n_layers-1] * 1.0;

//     for (int i = n_layers - 2; i >= 0; --i) {
//         imp_boundary[i] = imp_intrinsic[i] * (imp_boundary[i + 1] + imp_intrinsic[i] * tanh_un_thick[i]) /
//                           (imp_intrinsic[i] + imp_boundary[i + 1] * tanh_un_thick[i]);
//     }

//     tc::Tensor rtm = (imp_freespace - imp_boundary[0]) / (imp_freespace + imp_boundary[0]);
//     return rtm;
// }




// py::array_t<double> refl_transverse_electric(){

// }

