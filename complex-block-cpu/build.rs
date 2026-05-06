// build.rs
fn main() {
    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/openblas/lib");
    println!("cargo:rustc-link-lib=openblas");
}
