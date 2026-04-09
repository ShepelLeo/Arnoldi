# IRAM in Rust

Проект реализует `Implicitly Restarted Arnoldi Method (IRAM)` для вещественного несимметричного случая.
В рестарте используется exact-shift polynomial form, эквивалентная implicit QR restart, что позволяет сохранить реализацию в вещественной арифметике и не хранить лишние структуры.

## Что умеет

- матрица задаётся через `matvec`-оператор (`LinearOperator`), по умолчанию в CLI используется единичный оператор
- стартовый вектор можно подать из файла или сгенерировать случайно
- можно искать:
  - самые большие по модулю
  - самые маленькие по модулю
  - самые большие по значению (по действительной части)
  - самые маленькие по значению (по действительной части)
  - по половине с краёв спектра
- в выходной файл пишутся:
  - история сходимости по рестартам
  - количество итераций рестарта
  - число `matvec` операций
  - tracked peak heap memory
  - финальные Ritz-значения и оценки невязки

## Структура

- `src/operator.rs` — матвек-операторы и загрузка плотной матрицы
- `src/linalg/ops.rs` — базовые векторные операции
- `src/linalg/small.rs` — малый спектральный блок для матрицы Гессенберга
- `src/arnoldi.rs` — m-шаговый Arnoldi
- `src/selection.rs` — выбор нужных Ritz-значений и conjugate-pair handling
- `src/iram.rs` — основной цикл IRAM
- `src/report.rs` — текстовый отчёт о сходимости
- `src/memory.rs` — трекинг peak heap memory через global allocator

## Примеры запуска

Единичный оператор:

```bash
cargo run --release -- --dimension 32 --nev 1 --output iram_report.txt
```

Grcar-оператор:

```bash
cargo run --release -- --operator grcar --dimension 32 --nev 4 --ncv 14 --target largest-magnitude --output iram_report.txt
```

Плотная матрица из файла:

```bash
cargo run --release -- --matrix-file matrix.txt --nev 4 --ncv 14 --target largest-real --output iram_report.txt
```

Стартовый вектор из файла:

```bash
cargo run --release -- --matrix-file matrix.txt --start-vector v0.txt --nev 4 --ncv 14 --output iram_report.txt
```

## Формат входных файлов

- `matrix.txt`: квадратная матрица, одна строка на строку матрицы, значения разделены пробелами
- `v0.txt`: компоненты стартового вектора, разделённые пробелами или переводами строк

## Как задать свой matvec в коде

```rust
use ndarray::Array1;
use simple::{FnOperator, SolverConfig, SpectrumTarget, recommended_ncv, solve};

let n = 64;
let operator = FnOperator::new(n, "custom matvec", |x: &Array1<f64>| {
    Array1::from_iter((0..n).map(|i| {
        let left = if i > 0 { x[i - 1] } else { 0.0 };
        let right = x.get(i + 1).copied().unwrap_or(0.0);
        2.0 * x[i] - left + right
    }))
});

let start = Array1::ones(n);
let config = SolverConfig {
    nev: 2,
    ncv: recommended_ncv(2, n),
    max_restarts: 20,
    tol: 1.0e-10,
    breakdown_tol: 1.0e-12,
    target: SpectrumTarget::LargestMagnitude,
};

let report = solve(&operator, start, config, "ones start")?;
println!("{}", report.render_text());
# Ok::<(), simple::IramError>(())
```

## Замечание про единичную матрицу

Это корректный default по условию, но для одного стартового вектора Krylov-подпространство у `I` одномерно. Поэтому при `nev > 1` solver быстро попадёт в happy breakdown и честно сообщит, что из такого запуска нельзя извлечь больше независимых направлений.

## Convection-diffusion operator

Встроенный оператор `convection-diffusion` реализует центрально-разностную дискретизацию

```text
(Laplacian u) + rho * du/dx
```

на единичном квадрате `[0,1]x[0,1]` с нулевыми условиями Дирихле. Для него `--dimension` задаёт число внутренних узлов по одной оси `m`, а фактическая размерность матрицы равна `m*m`.

```bash
cargo run --release -- --operator convection-diffusion --dimension 20 --rho 100 --nev 4 --ncv 14 --target largest-real --output convdiff_report.txt
```

Формула `matvec` находится в `src/operator.rs` в `ConvectionDiffusionOperator::apply`.
