/*cppimport
<%
import os
import tectosaur
setup_pybind11(cfg)
cfg['compiler_args'] += ['-std=c++14', '-O3']
cfg['include_dirs'] += [os.path.join(tectosaur.source_dir, os.pardir)]
%>
*/

#include <utility>
#include <cmath>
#include <functional>
#include <iostream>
#include <cassert>

template <typename F, typename Fp>
std::pair<double,bool> newton(const F& f, const Fp& fp, double x0, double tol, int maxiter) {
    for (int i = 0; i < maxiter; i++) {
        double y = f(x0);
        double yp = fp(x0);
        double x1 = x0 - y / yp;
        // std::cout << x0 << " " << x1 << " " << y << " " << yp << x0 - x1 << std::endl;
        if (std::fabs(x1 - x0) <= tol * std::fabs(x0)) {
            return {x1, true};
        }
        x0 = x1;
    }
    return {x0, false}; 
}

double F(double V, double sigma_n, double state, double a, double V0) {
    return a * sigma_n * std::asinh(V / (2 * V0) * std::exp(state / a));
}

//https://www.wolframalpha.com/input/?i=d%5Ba*S*arcsinh(x+%2F+(2*y)+*+exp(s%2Fa))%5D%2Fdx
double dFdV(double V, double sigma_n, double state, double a, double V0) {
    double expsa = std::exp(state / a);
    double Q = (V * expsa) / (2 * V0);
    return a * expsa * sigma_n / (2 * V0 * std::sqrt(1 + (Q * Q)));
}

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "tectosaur/include/pybind11_nparray.hpp"
#include "tectosaur/include/vec_tensor.hpp"

namespace py = pybind11;

auto newton_py(std::function<double(double)> f,
        std::function<double(double)> fp,
        double x0, double tol, int maxiter) 
{
    return newton(f, fp, x0, tol, maxiter);
}

auto newton_rs(double tau_qs, double eta, double sigma_n,
        double state, double a, double V0, 
        double V_guess, double tol, int maxiter) 
{
    auto rsf = [&] (double V) { 
        return tau_qs - eta * V - F(V, sigma_n, state, a, V0); 
    };
    auto rsf_deriv = [&] (double V) { 
        return -eta - dFdV(V, sigma_n, state, a, V0); 
    };
    auto out = newton(rsf, rsf_deriv, V_guess, tol, maxiter);
    auto left = rsf(out.first * (1 - tol));
    auto right = rsf(out.first * (1 + tol));
    assert(left > out && right > out);
    return out;
}

void rate_state_solver(NPArray<double> tri_normals, NPArray<double> traction,
        NPArray<double> state, NPArray<double> velocity, NPArray<double> a,
        double eta, double V0,
        double additional_normal_stress,
        double tol, double maxiter)
{
    //TODO: check that as_ptr<Vec3> is working right.
    auto* tri_normals_ptr = as_ptr<Vec3>(tri_normals);
    auto* state_ptr = as_ptr<double>(state);
    auto* velocity_ptr = as_ptr<Vec3>(velocity);
    auto* traction_ptr = as_ptr<Vec3>(traction);
    auto* a_ptr = as_ptr<double>(a);

    size_t n_tris = tri_normals.request().shape[0];
    const int basis_dim = 3;
    const double eps = 1e-14;

    for (size_t i = 0; i < n_tris; i++) {
        auto normal = tri_normals_ptr[i];
        for (int d = 0; d < basis_dim; d++) {

            size_t dof = i * basis_dim + d;
            auto traction_vec = traction_ptr[dof];
            auto state = state_ptr[dof];

            auto normal_stress_vec = projection(traction_vec, normal);
            double normal_mag = length(normal_stress_vec);
            //TODO: remove when not in full space
            normal_mag += additional_normal_stress;
            auto shear_traction_vec = sub(traction_vec, normal_stress_vec);
            double shear_mag = length(shear_traction_vec);

            auto solve_result = newton_rs(
                shear_mag, eta, normal_mag, state,
                a_ptr[dof], V0, 0.0, tol, maxiter
            );
            assert(solve_result.second);
            double V_new_mag = solve_result.first;
            Vec3 V_new;
            if (shear_mag > eps) {
                auto shear_dir = div(shear_traction_vec, shear_mag);
                V_new = mult(shear_dir, V_new_mag);
                for (int d2 = 0; d2 < 3; d2++) {
                    velocity_ptr[dof][d2] = V_new[d2];
                }
            } else {
                for (int d2 = 0; d2 < 3; d2++) {
                    velocity_ptr[dof][d2] = 0.0;
                }
            }
        }
    }
}

PYBIND11_MODULE(qd_newton,m) {
    m.def("newton", &newton_py);
    m.def("newton_rs", &newton_rs);
    m.def("F", F);
    m.def("dFdV", dFdV);
    m.def("rate_state_solver", rate_state_solver);
}
