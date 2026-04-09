# IRAM in Rust

Пример из ARPACK
```bash
cargo run --release -- --operator convection-diffusion --dimension 10 --rho 0 --nev 4 --ncv 20 --max-restarts=500 --tol=1e-16 --target largest-magnitude --output convdiff_report.txt
```
