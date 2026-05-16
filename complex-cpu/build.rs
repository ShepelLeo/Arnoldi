// build.rs
fn main() {
    if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    } else {
        println!("cargo:rustc-link-search=native=/opt/ohpc/pub/libs/gnu15/openblas/0.3.30/lib");
    }

    println!("cargo:rustc-link-lib=dylib=openblas");
}