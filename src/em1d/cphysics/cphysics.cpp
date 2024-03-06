#include <torch/extension.h>
#include <torch/torch.h>
#include <cmath>
// #include <complex>

constexpr double mu0 = M_PI * 4e-7;
constexpr double epsilon0 = 8.854187817e-12;

torch::Tensor one_j = torch::complex(torch::zeros({1}), torch::ones({1}));

int working(int x){
    return x;
}

torch::Tensor calc_z(const torch::Tensor& sigma, const torch::Tensor& omega, const torch::Tensor& mu, const torch::Tensor& epsilon) {
    torch::Tensor z = one_j * mu * omega;
    return z;
}

torch::Tensor calc_y(const torch::Tensor& sigma, const torch::Tensor& omega, const torch::Tensor& mu, const torch::Tensor& epsilon) {
    torch::Tensor y = sigma + one_j * epsilon * omega;
    return y;
}

torch::Tensor calc_wavenumber(const torch::Tensor& sigma, const torch::Tensor& omega, const torch::Tensor& mu, const torch::Tensor& epsilon, bool sqrt = true) {
    torch::Tensor k = mu * omega * (epsilon * omega - one_j * sigma);
    if (sqrt) k = torch::sqrt(k);
    return k;
}

torch::Tensor freespace_u0(const torch::Tensor& omega, const torch::Tensor& wave_length, bool sqrt = true) {
    torch::Tensor k2 = mu0 * epsilon0 * omega.pow(2);
    torch::Tensor u02 = wave_length.pow(2) - k2;
    if (sqrt) u02 = torch::sqrt(u02);
    return u02;
}

torch::Tensor refl_transverse_electric(const torch::Tensor& sigma, const torch::Tensor& thickness, const torch::Tensor& omega, const torch::Tensor& wave_length, const torch::Tensor& mu, const torch::Tensor& epsilon) {

    int n_layers = sigma.numel();

    torch::Tensor lambda2 = wave_length.pow(2);

    torch::Tensor adm_boundary = torch::zeros({omega.numel(), wave_length.numel() }, torch::kComplexDouble);

    torch::Tensor adm_freespace = torch::sqrt(lambda2 - omega.pow(2) * mu0 * epsilon0) / (one_j * omega * mu0);

    torch::Tensor k2 = calc_wavenumber(sigma, omega, mu, epsilon, false);
    torch::Tensor zn = calc_z(sigma, omega, mu, epsilon);

    torch::Tensor adm_intrinsic = torch::sqrt(lambda2 - k2);
    torch::Tensor tanh_un_thick = torch::tanh(adm_intrinsic.slice(0, 0, n_layers - 1) * thickness);
    adm_intrinsic /= zn;

    adm_boundary = adm_intrinsic[n_layers-1];

    for (int i = n_layers - 2; i >= 0; --i) {
        adm_boundary = adm_intrinsic[i] * (adm_boundary + adm_intrinsic[i] * tanh_un_thick[i]) /
                          (adm_intrinsic[i] + adm_boundary * tanh_un_thick[i]);
    }

    torch::Tensor rte = (adm_freespace - adm_boundary) / (adm_freespace + adm_boundary);
    return rte;
}

// torch::Tensor refl_transverse_magnetic(const torch::Tensor& sigma, const torch::Tensor& thickness, const torch::Tensor& omega, const torch::Tensor& wave_length, double mu = mu0, double epsilon = epsilon0) {
//     torch::Tensor mu_tensor = torch::ones_like(sigma) * mu;
//     torch::Tensor epsilon_tensor = torch::ones_like(sigma) * epsilon;

//     int n_layers = sigma.numel();

//     torch::Tensor lambda2 = wave_length.pow(2);

//     torch::Tensor imp_boundary  = torch::zeros({ sigma.numel(), omega.numel(), wave_length.numel() }, torch::kComplexDouble);
//     torch::Tensor imp_freespace = torch::sqrt(lambda2 - omega.pow(2) * mu0 * epsilon0) / torch::complex<double>(0, omega * mu0);

//     torch::Tensor k2 = calc_wavenumber(sigma, omega, mu, epsilon, false);
//     torch::Tensor yn = calc_y(sigma, omega, mu, epsilon);

//     torch::Tensor imp_intrinsic = torch::sqrt(lambda2 - k2);
//     torch::Tensor tanh_un_thick = torch::tanh(imp_intrinsic.slice(0, 0, n_layers - 1) * thickness);
//     imp_intrinsic = imp_intrinsic / yn;

//     imp_boundary[n_layers-1] = imp_intrinsic[n_layers-1] * 1.0;

//     for (int i = n_layers - 2; i >= 0; --i) {
//         imp_boundary[i] = imp_intrinsic[i] * (imp_boundary[i + 1] + imp_intrinsic[i] * tanh_un_thick[i]) /
//                           (imp_intrinsic[i] + imp_boundary[i + 1] * tanh_un_thick[i]);
//     }

//     torch::Tensor rtm = (imp_freespace - imp_boundary[0]) / (imp_freespace + imp_boundary[0]);
//     return rtm;
// }


PYBIND11_MODULE(cphysics, m) {
  m.def("working", &working, "test if working");
  m.def("freespace_u0", &freespace_u0, "test if working");
  m.def("refl_transverse_electric2", &refl_transverse_electric, "test if working");
}
