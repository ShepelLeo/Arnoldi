use std::path::PathBuf;

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

        let nvcc = find_nvcc();
        let mut cuda_build = cc::Build::new();

        cuda_build
            .cuda(true)
            .no_default_flags(true)
            .warnings(false)
            .compiler(&nvcc)
            .flag("-O3")
            .flag("-std=c++17")
            .flag("--compiler-options=-fPIC")
            .file("src/linalg/magma_shifted_qr.cu");

        if let Ok(arch) = std::env::var("CUDA_ARCH") {
            cuda_build.flag(&format!("-arch={arch}"));
        }

        cuda_build.compile("complex_iram_magma_shifted_qr");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cusparse");
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
}


fn find_nvcc() -> PathBuf {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_DIR");
    println!("cargo:rerun-if-env-changed=EBROOTCUDA");

    if let Some(path) = std::env::var_os("NVCC") {
        return PathBuf::from(path);
    }

    for var in ["CUDA_HOME", "CUDA_DIR", "EBROOTCUDA"] {
        if let Some(root) = std::env::var_os(var) {
            let candidate = PathBuf::from(root).join("bin").join("nvcc");
            if candidate.exists() {
                return candidate;
            }
        }
    }

    PathBuf::from("nvcc")
}
