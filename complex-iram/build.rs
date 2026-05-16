fn main() {
    if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    } else {
        println!("cargo:rustc-link-search=native=/opt/ohpc/pub/libs/gnu15/openblas/0.3.30/lib");
    }
    println!("cargo:rustc-link-lib=dylib=openblas");

    if std::env::var_os("CARGO_FEATURE_MAGMA").is_some() {
        if let Ok(dir) = std::env::var("MAGMA_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-lib=dylib=magma");
        } else if let Ok(dir) = std::env::var("MAGMA_DIR") {
            println!("cargo:rustc-link-search=native={dir}/lib");
            println!("cargo:rustc-link-lib=dylib=magma");
        } else {
            panic!(
                "MAGMA backend enabled but MAGMA was not found: set MAGMA_LIB_DIR=/path/to/magma/lib or MAGMA_DIR=/path/to/magma"
            );
        }

        if let Ok(dir) = std::env::var("CUDA_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }

        if let Ok(dir) = std::env::var("CUBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }

        if let Ok(dir) = std::env::var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={dir}/lib64");
            println!("cargo:rustc-link-search=native={dir}/targets/x86_64-linux/lib");
        }

        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}
