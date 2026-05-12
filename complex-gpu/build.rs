fn main() {
    if let Ok(dir) = std::env::var("MAGMA_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    } else if let Ok(dir) = std::env::var("MAGMA_DIR") {
        println!("cargo:rustc-link-search=native={dir}/lib");
    }

    if let Ok(dir) = std::env::var("CUDA_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    } else if let Ok(dir) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={dir}/lib64");
    }

    println!("cargo:rustc-link-lib=magma");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudart");
}
