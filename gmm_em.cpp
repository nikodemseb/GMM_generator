// gmm_em.cpp
// A teaching-focused Gaussian Mixture Model (GMM) implementation in modern C++17
// Demonstrates: E-step/M-step pattern, log-likelihood monitoring, and cluster interpretation.
// Single-file, no third-party deps. Compiles with: g++ -O2 -std=c++17 gmm_em.cpp -o gmm
// Usage examples:
//   ./gmm --k 3 --max_iters 200 --tol 1e-6                # fit on synthetic 2D data
//   ./gmm --input data.csv --k 4 --save out.csv           # fit on CSV and save responsibilities/labels
// CSV: rows are samples, columns are numeric features (no header expected).

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using std::size_t;

// -------- Small matrix helpers (row-major) --------
struct Matrix {
    size_t r{0}, c{0};
    std::vector<double> a; // row-major r x c
    Matrix() = default;
    Matrix(size_t r_, size_t c_, double v = 0.0) : r(r_), c(c_), a(r_*c_, v) {}
    inline double& operator()(size_t i, size_t j) { return a[i*c + j]; }
    inline const double& operator()(size_t i, size_t j) const { return a[i*c + j]; }
    static Matrix eye(size_t d, double s = 1.0) {
        Matrix I(d, d, 0.0); for (size_t i = 0; i < d; ++i) I(i,i) = s; return I; }
};

struct Vector {
    std::vector<double> v;
    Vector() = default;
    explicit Vector(size_t n, double val = 0.0) : v(n, val) {}
    inline size_t size() const { return v.size(); }
    inline double& operator[](size_t i) { return v[i]; }
    inline const double& operator[](size_t i) const { return v[i]; }
};

static inline double dot(const Vector& x, const Vector& y) {
    double s = 0.0; for (size_t i = 0; i < x.size(); ++i) s += x.v[i] * y.v[i]; return s; }

static inline Vector sub(const Vector& a, const Vector& b) {
    Vector r(a.size()); for (size_t i=0;i<a.size();++i) r[i] = a.v[i] - b.v[i]; return r; }

static inline Vector col(const Matrix& X, size_t j) {
    Vector r(X.r); for (size_t i=0;i<X.r;++i) r[i] = X(i,j); return r; }

static inline void add_outer(Matrix& S, const Vector& x, double w = 1.0) {
    // S += w * x x^T
    size_t d = S.r; for (size_t i=0;i<d;++i) for (size_t j=0;j<d;++j) S(i,j) += w * x.v[i] * x.v[j]; }

// Cholesky decomposition: A = L L^T (A SPD). Returns true if success.
bool cholesky(const Matrix& A, Matrix& L) {
    size_t n = A.r; L = Matrix(n,n,0.0);
    for (size_t i=0;i<n;++i) {
        for (size_t j=0;j<=i;++j) {
            double s = A(i,j);
            for (size_t k=0;k<j;++k) s -= L(i,k)*L(j,k);
            if (i==j) {
                if (s <= 0.0) return false; // not SPD
                L(i,j) = std::sqrt(s);
            } else {
                L(i,j) = s / L(j,j);
            }
        }
    }
    return true;
}

// Solve L y = b for y (forward substitution), L lower-triangular
static inline void forward_solve(const Matrix& L, const Vector& b, Vector& y) {
    size_t n = L.r; y = Vector(n);
    for (size_t i=0;i<n;++i) {
        double s = b.v[i];
        for (size_t k=0;k<i;++k) s -= L(i,k) * y[k];
        y[i] = s / L(i,i);
    }
}

// Log multivariate Gaussian pdf using Cholesky (stable)
static inline double log_gaussian_pdf(const Vector& x, const Vector& mu, const Matrix& Sigma, double* out_sq = nullptr) {
    const size_t d = mu.size();
    // Try Cholesky with jitter if needed
    Matrix S = Sigma, L; // copy
    // ensure symmetry (guard against drift)
    for (size_t i=0;i<d;++i) for (size_t j=i+1;j<d;++j) S(i,j) = S(j,i) = 0.5*(Sigma(i,j)+Sigma(j,i));

    const double base = 1e-6; bool ok=false; double jitter = 0.0; int tries=0;
    while (!ok && tries < 6) {
        if (jitter>0.0) for (size_t i=0;i<d;++i) S(i,i) += jitter; // add to diagonal
        ok = cholesky(S, L);
        if (!ok) { jitter = (jitter==0.0? base : jitter*10.0); ++tries; for (size_t i=0;i<d;++i) S(i,i) = Sigma(i,i); /* reset diag before next add */ }
    }
    if (!ok) {
        // fall back to large diagonal regularization
        S = Sigma; for (size_t i=0;i<d;++i) { for (size_t j=0;j<d;++j) S(i,j) = 0.0; S(i,i) = 1.0; }
        cholesky(S, L);
    }

    Vector v = sub(x, mu);
    Vector y; forward_solve(L, v, y);
    double sq = dot(y, y); if (out_sq) *out_sq = sq;
    double sumlog = 0.0; for (size_t i=0;i<d;++i) sumlog += std::log(L(i,i));
    const double log2pi = std::log(2.0 * M_PI);
    return -0.5 * d * log2pi - sumlog - 0.5 * sq;
}

struct GMM {
    size_t k{2};           // components
    size_t d{0};           // dimensionality
    size_t n{0};           // samples
    double tol{1e-5};
    size_t max_iters{200};
    double min_covar{1e-6};
    uint64_t seed{42};

    // Parameters
    std::vector<double> weights;    // size k
    std::vector<Vector> means;      // k x d
    std::vector<Matrix> covs;       // k x dxd

    // Responsibilities
    Matrix resp; // n x k (filled in E-step)

    // Initialize using k-means++ style means, uniform weights, shared covariance
    void init(const Matrix& X) {
        n = X.r; d = X.c;
        std::mt19937_64 rng(seed);

        // Compute global mean and covariance (for initial covariances)
        Vector global_mean(d,0.0);
        for (size_t i=0;i<n;++i) for (size_t j=0;j<d;++j) global_mean[j] += X(i,j);
        for (size_t j=0;j<d;++j) global_mean[j] /= static_cast<double>(n);
        Matrix global_cov(d,d,0.0);
        for (size_t i=0;i<n;++i) {
            Vector xi(d); for (size_t j=0;j<d;++j) xi[j] = X(i,j);
            Vector v = sub(xi, global_mean);
            add_outer(global_cov, v, 1.0);
        }
        for (size_t i=0;i<d;++i) for (size_t j=0;j<d;++j) global_cov(i,j) /= std::max<double>(1.0, n-1.0);
        for (size_t i=0;i<d;++i) global_cov(i,i) += min_covar; // regularize

        // k-means++ seeding of means
        std::uniform_int_distribution<size_t> unif_idx(0, n-1);
        std::vector<size_t> centers; centers.reserve(k);
        centers.push_back(unif_idx(rng));
        std::vector<double> d2(n, std::numeric_limits<double>::infinity());
        auto sqdist = [&](size_t i, const Vector& mu){ double s=0; for(size_t j=0;j<d;++j){ double t=X(i,j)-mu.v[j]; s+=t*t; } return s; };
        auto idx_to_vec = [&](size_t i){ Vector v(d); for(size_t j=0;j<d;++j) v[j]=X(i,j); return v; };
        while (centers.size() < k) {
            Vector last = idx_to_vec(centers.back());
            for (size_t i=0;i<n;++i) d2[i] = std::min(d2[i], sqdist(i, last));
            std::discrete_distribution<size_t> dist(d2.begin(), d2.end());
            centers.push_back(dist(rng));
        }

        weights.assign(k, 1.0/static_cast<double>(k));
        means.clear(); means.reserve(k);
        covs.clear(); covs.reserve(k);
        for (size_t t=0;t<k;++t) {
            Vector mu = idx_to_vec(centers[t]);
            means.push_back(mu);
            covs.push_back(global_cov); // start with shared covariance
        }
        resp = Matrix(n, k, 0.0);
    }

    // E-step: compute responsibilities and log-likelihood
    double e_step(const Matrix& X) {
        double ll = 0.0; // log-likelihood
        for (size_t i=0;i<n;++i) {
            std::vector<double> logp(k);
            for (size_t t=0;t<k;++t) {
                Vector xi(d); for (size_t j=0;j<d;++j) xi[j] = X(i,j);
                double lg = log_gaussian_pdf(xi, means[t], covs[t]);
                logp[t] = std::log(std::max(weights[t], 1e-16)) + lg;
            }
            // log-sum-exp
            double m = *std::max_element(logp.begin(), logp.end());
            double sum = 0.0; for (size_t t=0;t<k;++t) sum += std::exp(logp[t] - m);
            double logsumexp = m + std::log(sum);
            ll += logsumexp;
            for (size_t t=0;t<k;++t) resp(i,t) = std::exp(logp[t] - logsumexp);
        }
        return ll;
    }

    // M-step: update weights, means, covariances
    void m_step(const Matrix& X) {
        std::vector<double> Nk(k, 0.0);
        for (size_t t=0;t<k;++t) for (size_t i=0;i<n;++i) Nk[t] += resp(i,t);

        // Update weights
        for (size_t t=0;t<k;++t) weights[t] = std::max(Nk[t] / static_cast<double>(n), 1e-16);
        // normalize (safety)
        double s = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (double& w : weights) w /= s;

        // Update means
        for (size_t t=0;t<k;++t) {
            Vector mu(d,0.0);
            if (Nk[t] <= 1e-10) {
                // re-seed dead component to a random point responsibility-wise (highest unexplained)
                size_t idx = t % n; // deterministic simple fallback
                for (size_t j=0;j<d;++j) mu[j] = X(idx,j);
            } else {
                for (size_t i=0;i<n;++i) for (size_t j=0;j<d;++j) mu[j] += resp(i,t) * X(i,j);
                for (size_t j=0;j<d;++j) mu[j] /= Nk[t];
            }
            means[t] = mu;
        }

        // Update covariances
        for (size_t t=0;t<k;++t) {
            Matrix S(d,d,0.0);
            if (Nk[t] <= 1e-10) {
                // reset to identity if dead
                for (size_t i=0;i<d;++i) S(i,i) = 1.0;
            } else {
                for (size_t i=0;i<n;++i) {
                    Vector xi(d); for (size_t j=0;j<d;++j) xi[j] = X(i,j);
                    Vector v = sub(xi, means[t]);
                    add_outer(S, v, resp(i,t));
                }
                for (size_t i=0;i<d;++i) for (size_t j=0;j<d;++j) S(i,j) /= Nk[t];
                // regularize diagonal
                for (size_t i=0;i<d;++i) S(i,i) += min_covar;
            }
            covs[t] = S;
        }
    }

    // Fit model via EM. Returns final log-likelihood and number of iterations.
    std::pair<double,size_t> fit(const Matrix& X) {
        init(X);
        double prev_ll = -std::numeric_limits<double>::infinity();
        size_t it = 0;
        for (; it < max_iters; ++it) {
            double ll = e_step(X);
            std::cout << "iter " << std::setw(3) << it+1 << ": loglik = " << std::setprecision(12) << ll << "\n";
            m_step(X);
            if (std::abs(ll - prev_ll) < tol * (1.0 + std::abs(prev_ll))) {
                prev_ll = ll; break; // converged
            }
            prev_ll = ll;
        }
        if (it==max_iters) std::cout << "Reached max_iters without strict convergence.\n";
        return {prev_ll, it+1};
    }

    // Predict hard labels (argmax responsibility) and optionally return responsibilities
    std::vector<size_t> predict(const Matrix& X, Matrix* out_resp = nullptr) {
        size_t N = X.r; Matrix R(N,k,0.0);
        // reuse e-step machinery without changing internal state
        for (size_t i=0;i<N;++i) {
            std::vector<double> logp(k);
            for (size_t t=0;t<k;++t) {
                Vector xi(d); for (size_t j=0;j<d;++j) xi[j] = X(i,j);
                double lg = log_gaussian_pdf(xi, means[t], covs[t]);
                logp[t] = std::log(std::max(weights[t], 1e-16)) + lg;
            }
            double m = *std::max_element(logp.begin(), logp.end());
            double sum = 0.0; for (size_t t=0;t<k;++t) sum += std::exp(logp[t] - m);
            double lse = m + std::log(sum);
            for (size_t t=0;t<k;++t) R(i,t) = std::exp(logp[t] - lse);
        }
        if (out_resp) *out_resp = R;
        std::vector<size_t> labels(N,0);
        for (size_t i=0;i<N;++i) {
            double best = -1.0; size_t arg=0;
            for (size_t t=0;t<k;++t) if (R(i,t) > best) { best = R(i,t); arg = t; }
            labels[i] = arg;
        }
        return labels;
    }
};

// -------- Utility: synthetic data generator (2D mixture) --------
Matrix sample_synthetic(size_t N, uint64_t seed=123) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> n01(0.0,1.0);
    Matrix X(N, 2, 0.0);
    // two clusters with different covariances
    Matrix L1(2,2,0.0); L1(0,0)=1.0; L1(1,0)=0.6; L1(1,1)=0.8; // covariance via L L^T
    Matrix L2(2,2,0.0); L2(0,0)=0.5; L2(1,0)=-0.2; L2(1,1)=0.6;
    Vector m1(2); m1[0]=-3.0; m1[1]=0.0;
    Vector m2(2); m2[0]=+3.0; m2[1]=+2.0;

    std::bernoulli_distribution choose(0.5);
    for (size_t i=0;i<N;++i) {
        bool c = choose(rng);
        double z0 = n01(rng), z1 = n01(rng);
        Vector z(2); z[0]=z0; z[1]=z1;
        Vector x(2);
        if (!c) {
            x[0] = m1[0] + L1(0,0)*z[0];
            x[1] = m1[1] + L1(1,0)*z[0] + L1(1,1)*z[1];
        } else {
            x[0] = m2[0] + L2(0,0)*z[0];
            x[1] = m2[1] + L2(1,0)*z[0] + L2(1,1)*z[1];
        }
        X(i,0)=x[0]; X(i,1)=x[1];
    }
    return X;
}

// --------- CSV I/O ---------
std::optional<Matrix> read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;
    std::vector<std::vector<double>> rows; rows.reserve(1024);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell; std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            try { row.push_back(std::stod(cell)); }
            catch(...) { /* non-numeric cell -> skip line */ row.clear(); break; }
        }
        if (!row.empty()) rows.push_back(std::move(row));
    }
    if (rows.empty()) return std::nullopt;
    size_t d = rows[0].size();
    Matrix X(rows.size(), d, 0.0);
    for (size_t i=0;i<rows.size();++i) if (rows[i].size()==d) for (size_t j=0;j<d;++j) X(i,j) = rows[i][j];
    return X;
}

void write_csv(const std::string& path, const Matrix& X, const Matrix& R, const std::vector<size_t>& labels) {
    std::ofstream out(path);
    if (!out) { std::cerr << "Could not write to " << path << "\n"; return; }
    // header
    out << "label";
    for (size_t t=0;t<R.c;++t) out << ",resp_" << t;
    for (size_t j=0;j<X.c;++j) out << ",x" << j;
    out << "\n";

    for (size_t i=0;i<X.r;++i) {
        out << labels[i];
        for (size_t t=0;t<R.c;++t) out << "," << std::setprecision(10) << R(i,t);
        for (size_t j=0;j<X.c;++j) out << "," << std::setprecision(10) << X(i,j);
        out << "\n";
    }
    std::cout << "Saved predictions to " << path << "\n";
}

// --------- CLI parsing ---------
struct Args {
    std::string input=""; std::string save=""; size_t k=2; size_t max_iters=200; double tol=1e-5; uint64_t seed=42; double min_covar=1e-6; size_t Nsynthetic=600; };

Args parse_args(int argc, char** argv) {
    Args a; for (int i=1;i<argc;++i) {
        std::string s(argv[i]);
        auto need = [&](int i){ if (i+1>=argc) { std::cerr << "Missing value after " << s << "\n"; std::exit(2);} return std::string(argv[i+1]); };
        if (s=="--input") { a.input = need(i); ++i; }
        else if (s=="--save") { a.save = need(i); ++i; }
        else if (s=="--k") { a.k = static_cast<size_t>(std::stoul(need(i))); ++i; }
        else if (s=="--max_iters") { a.max_iters = static_cast<size_t>(std::stoul(need(i))); ++i; }
        else if (s=="--tol") { a.tol = std::stod(need(i)); ++i; }
        else if (s=="--seed") { a.seed = static_cast<uint64_t>(std::stoull(need(i))); ++i; }
        else if (s=="--min_covar") { a.min_covar = std::stod(need(i)); ++i; }
        else if (s=="--n") { a.Nsynthetic = static_cast<size_t>(std::stoul(need(i))); ++i; }
        else if (s=="-h" || s=="--help") {
            std::cout << "GMM via EM (C++17)\n"
                         "Usage: ./gmm [--input data.csv] [--k 3] [--max_iters 200] [--tol 1e-5] [--seed 42] [--min_covar 1e-6] [--save out.csv] [--n 600]\n";
            std::exit(0);
        }
    }
    return a;
}

// --------- Pretty printing for interpretation ---------
void print_vector(const Vector& v, const std::string& name) {
    std::cout << name << " = [";
    for (size_t i=0;i<v.size();++i) { if (i) std::cout << ", "; std::cout << std::setprecision(6) << v.v[i]; }
    std::cout << "]\n";
}

void print_matrix(const Matrix& M, const std::string& name) {
    std::cout << name << " =\n";
    for (size_t i=0;i<M.r;++i) {
        std::cout << "  [";
        for (size_t j=0;j<M.c;++j) {
            if (j) std::cout << ", ";
            std::cout << std::setw(10) << std::setprecision(6) << M(i,j);
        }
        std::cout << "]\n";
    }
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    Args args = parse_args(argc, argv);

    Matrix X;
    if (!args.input.empty()) {
        auto maybe = read_csv(args.input);
        if (!maybe) { std::cerr << "Failed to read dataset from '" << args.input << "'\n"; return 1; }
        X = *maybe;
        std::cout << "Loaded " << X.r << " rows x " << X.c << " cols from '" << args.input << "'\n";
    } else {
        X = sample_synthetic(args.Nsynthetic, args.seed);
        std::cout << "Generated synthetic dataset: " << X.r << " x " << X.c << " (2D)\n";
    }

    GMM model; model.k = args.k; model.max_iters = args.max_iters; model.tol = args.tol; model.seed = args.seed; model.min_covar = args.min_covar;
    model.fit(X);

    // Report learned parameters (cluster interpretation)
    std::cout << "\nMixture weights (cluster proportions):\n";
    for (size_t t=0;t<model.k;++t) std::cout << "  pi[" << t << "] = " << std::setprecision(6) << model.weights[t] << "\n";
    for (size_t t=0;t<model.k;++t) { print_vector(model.means[t], "mu[" + std::to_string(t) + "]"); }
    for (size_t t=0;t<model.k;++t) { print_matrix(model.covs[t],  "Sigma[" + std::to_string(t) + "]"); }

    // Predict labels and responsibilities
    Matrix R; auto labels = model.predict(X, &R);

    // Optional save
    if (!args.save.empty()) {
        write_csv(args.save, X, R, labels);
    }

    // Quick confusion-like summary: proportion of points most assigned to each cluster
    std::vector<size_t> counts(model.k,0);
    for (size_t i=0;i<X.r;++i) counts[labels[i]]++;
    std::cout << "\nHard assignment counts (for interpretation):\n";
    for (size_t t=0;t<model.k;++t) std::cout << "  cluster " << t << ": " << counts[t] << " (" << std::setprecision(4) << (100.0*counts[t]/(double)X.r) << "%)\n";

    std::cout << "\nTip: Visualize 2D results by plotting points colored by label and overlaying covariance ellipses (eigenvectors/values of Sigma_k).\n";

    return 0;
}
