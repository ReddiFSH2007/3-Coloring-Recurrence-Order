#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <iomanip>
#include <ext/pb_ds/assoc_container.hpp>

using namespace std;
using namespace __gnu_pbds;

const uint32_t MOD = 2147483647; 

struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};

long long power(long long base, long long exp) {
    long long res = 1;
    base %= MOD;
    while (exp > 0) {
        if (exp & 1) res = (res * base) % MOD;
        base = (base * base) % MOD;
        exp >>= 1;
    }
    return res;
}

long long modInverse(long long n) { return power(n, MOD - 2); }

inline uint64_t get_canonical(uint64_t state, int m, int bits) {
    int mapping[32]; 
    for(int i=0; i<32; ++i) mapping[i] = -1;
    int next_color = 0;
    uint64_t canonical = 0;
    uint64_t mask = (1ULL << bits) - 1;
    for (int i = m - 1; i >= 0; --i) {
        int c = (state >> (i * bits)) & mask;
        if (mapping[c] == -1) mapping[c] = next_color++;
        canonical = (canonical << bits) | mapping[c];
    }
    return canonical;
}

void init_dfs_layer0(int idx, int max_c, uint64_t curr, int m, int q, int bits, long long w_same,
                     gp_hash_table<uint64_t, int, custom_hash>& canonical_to_id, 
                     vector<uint64_t>& unique_states,
                     vector<uint32_t>& v_init) {
    if (idx == m) {
        uint64_t cid = get_canonical(curr, m, bits);

        long long weight = 1;
        for (int i = 0; i <= max_c; ++i) {
            weight = (weight * (q - i)) % MOD;
        }
        
 
        long long internal_weight = 1;
        uint64_t mask = (1ULL << bits) - 1;
        for (int i = 0; i < m - 1; ++i) {
            int c1 = (curr >> ((m - 1 - i) * bits)) & mask;
            int c2 = (curr >> ((m - 1 - (i + 1)) * bits)) & mask;
            if (c1 == c2) internal_weight = (internal_weight * w_same) % MOD;
        }
        weight = (weight * internal_weight) % MOD;

 
        if (weight == 0) return; 

        auto it = canonical_to_id.find(cid);
        if (it == canonical_to_id.end()) {
            canonical_to_id[cid] = unique_states.size();
            unique_states.push_back(cid);
            v_init.push_back(weight);
        } else {
            v_init[it->second] = (v_init[it->second] + weight) % MOD;
        }
        return;
    }
    int limit = min(q - 1, max_c + 1);
    for (int c = 0; c <= limit; ++c) {
        uint64_t next_curr = curr | ((uint64_t)c << ((m - 1 - idx) * bits));
        init_dfs_layer0(idx + 1, max(max_c, c), next_curr, m, q, bits, w_same, canonical_to_id, unique_states, v_init);
    }
}

void solve(int m, int q, long long w_same, int MAX_TERMS) {
    int bits = 0;
    while ((1ULL << bits) < (uint64_t)q) bits++;
    
    if (m * bits > 64) {
        cout << "Error: m * bits > 64" << endl;
        return;
    }

    auto start_time = chrono::high_resolution_clock::now();
    vector<uint64_t> layer_states[22];
    gp_hash_table<uint64_t, int, custom_hash> layer_map[22];
    vector<uint32_t> csr_row_ptr[22], csr_col_idx[22], csr_val[22]; 
    vector<uint32_t> v_init;

    init_dfs_layer0(0, 0, 0, m, q, bits, w_same, layer_map[0], layer_states[0], v_init);

    uint64_t mask = (1ULL << bits) - 1;

    for (int row = 0; row < m; ++row) {
        int target_layer = (row == m - 1) ? 0 : row + 1;
        vector<vector<pair<uint32_t, uint32_t>>> temp_adj; 

        for (int i = 0; i < (int)layer_states[row].size(); ++i) {
            uint64_t state = layer_states[row][i]; 
            int c_old = (state >> ((m - 1 - row) * bits)) & mask;
            int c_up = (row == 0) ? -1 : ((state >> ((m - 1 - (row - 1)) * bits)) & mask);

            for (int c_new = 0; c_new < q; ++c_new) {
                long long t_weight = 1;
                if (c_new == c_old) t_weight = (t_weight * w_same) % MOD;
                if (c_new == c_up) t_weight = (t_weight * w_same) % MOD;
                
                if (t_weight == 0) continue; 

                uint64_t next_state = state & ~(mask << ((m - 1 - row) * bits)); 
                next_state |= ((uint64_t)c_new << ((m - 1 - row) * bits)); 
                uint64_t can_next = get_canonical(next_state, m, bits); 

                int next_id;
                auto it = layer_map[target_layer].find(can_next);
                if (it == layer_map[target_layer].end()) {
                    next_id = layer_states[target_layer].size();
                    layer_map[target_layer][can_next] = next_id;
                    layer_states[target_layer].push_back(can_next);
                } else {
                    next_id = it->second;
                }
                while(temp_adj.size() <= next_id) temp_adj.push_back({});
                temp_adj[next_id].push_back({i, (uint32_t)t_weight});
            }
        }

        int next_size = layer_states[target_layer].size();
        csr_row_ptr[row].assign(next_size + 1, 0);
        size_t total_edges = 0;
        for(auto& adj : temp_adj) total_edges += adj.size();
        csr_col_idx[row].reserve(total_edges);
        csr_val[row].reserve(total_edges);

        for (int i = 0; i < next_size; ++i) {
            csr_row_ptr[row][i] = csr_col_idx[row].size();
            if (i < (int)temp_adj.size()) {
                for (auto& edge : temp_adj[i]) {
                    csr_col_idx[row].push_back(edge.first);
                    csr_val[row].push_back(edge.second);
                }
            }
        }
        csr_row_ptr[row][next_size] = csr_col_idx[row].size();
        csr_col_idx[row].shrink_to_fit(); 
        csr_val[row].shrink_to_fit();

        if (row != 0) layer_map[row].clear();
    }

    int num_initial_states = layer_states[0].size();
    mt19937 rng(random_device{}());
    uniform_int_distribution<uint32_t> dist(1, MOD - 1);
    vector<uint32_t> random_weights(num_initial_states);
    for (int i = 0; i < num_initial_states; ++i) random_weights[i] = dist(rng);

    vector<vector<uint32_t>> v_layers(m + 1);
    for (int row = 0; row <= m; ++row) {
        v_layers[row].resize(layer_states[row == m ? 0 : row].size(), 0);
    }
    for (int i = 0; i < num_initial_states; ++i) v_layers[0][i] = v_init[i];

    
    vector<long long> sequence;
    sequence.reserve(MAX_TERMS);
    vector<long long> C = {1}, B = {1};
    C.reserve((MAX_TERMS / 2) + 100); 
    B.reserve((MAX_TERMS / 2) + 100); 
    
    int L = 0, m_bm = 1;
    long long b = 1;

    for (int n = 0; n < MAX_TERMS; ++n) {
        uint64_t current_sum = 0;
        const uint32_t* v_l0 = v_layers[0].data();
        const uint32_t* rw = random_weights.data();
        for (int i = 0; i < num_initial_states; ++i) {
            current_sum = (current_sum + (uint64_t)v_l0[i] * rw[i]) % MOD;
        }
        sequence.push_back(current_sum);

        for (int row = 0; row < m; ++row) {
            int next_sz = layer_states[row == m - 1 ? 0 : row + 1].size();
            const uint32_t* r_ptr = csr_row_ptr[row].data();
            const uint32_t* c_idx = csr_col_idx[row].data();
            const uint32_t* c_val_arr = csr_val[row].data();
            const uint32_t* v_in = v_layers[row].data();
            uint32_t* v_out = v_layers[row + 1].data();

            #pragma omp parallel for schedule(static, 512)
            for (int i = 0; i < next_sz; ++i) {
                uint32_t start = r_ptr[i];
                uint32_t end = r_ptr[i + 1];
                uint64_t s = 0;
                for (uint32_t p = start; p < end; ++p) {
                    // Multiply state values by the thermal transfer matrix elements
                    s += ((uint64_t)v_in[c_idx[p]] * c_val_arr[p]);
                }
                v_out[i] = s % MOD;
            }
        }
        v_layers[0].swap(v_layers[m]); 

        long long d = 0;
        if (L > 2000) {
            long long d_shared = 0;
            #pragma omp parallel
            {
                unsigned long long d_local = 0;
                #pragma omp for schedule(static, 2048)
                for (int j = 0; j <= L; ++j) {
                    d_local += (unsigned long long)C[j] * sequence[n - j];
                    if (d_local >= 1700000000000000000ULL) d_local %= MOD;
                }
                #pragma omp critical
                { d_shared = (d_shared + d_local) % MOD; }
            }
            d = d_shared;
        } else {
            unsigned long long d_fast = 0;
            for (int j = 0; j <= L; ++j) {
                d_fast += (unsigned long long)C[j] * sequence[n - j];
                if (d_fast >= 1700000000000000000ULL) d_fast %= MOD;
            }
            d = d_fast % MOD;
        }

        if (d == 0) {
            m_bm++;
        } else {
            vector<long long> T = C;
            long long c_val = (d * modInverse(b)) % MOD;
            while (C.size() <= B.size() + m_bm) C.push_back(0);
            
            #pragma omp parallel for schedule(static, 2048) if(B.size() > 2000)
            for (int j = 0; j < (int)B.size(); ++j) {
                C[j + m_bm] = (C[j + m_bm] - c_val * B[j]) % MOD;
                if (C[j + m_bm] < 0) C[j + m_bm] += MOD;
            }
            
            if (2 * L <= n) { 
                L = n + 1 - L; 
                B = T; 
                b = d; 
                m_bm = 1; 
            } else { 
                m_bm++; 
            }
        }

    }

    auto end_time = chrono::high_resolution_clock::now();
    cout << L << " ";
}

long long floatToModInt(float w) {
    if (w <= 0.0001f) return 0; 
    if (w >= 0.9999f) return 1;

    long long num = round(w * 10.0f);
    long long den = 10;

    return (num * modInverse(den)) % MOD;
}

int main() {         
    // int q = 3;            
    int d[] = {1, 1, 2, 4, 10, 26, 76, 232, 750, 2494, 8524, 29624, 104468}; 
    // theoretical minimum degree for q>=4, and of course greater than q=2 and 3;
    
    float w_float = 0.1f;
    long long w_mod = floatToModInt(w_float);
    
    cout << fixed << setprecision(1) << "w = " << w_float << " (mod " << w_mod << ")" << endl;
    
    for (int q = 2; q <= 6; q++) {
        cout << "q = " << q << " : ";
        for (int m = 1; m < 12; ++m) {
            solve(m, q, w_mod, 2 * d[m] + 50); 
        }
        cout << endl;
    }
    
    return 0;
}
