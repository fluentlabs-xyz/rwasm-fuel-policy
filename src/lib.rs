#![cfg_attr(not(feature = "std"), no_std)]

//! Fuel metering primitives used by the rWasm / Wasm runtime.
//!
//! This module provides:
//! - **Per-operator fuel costs** for `wasmparser::Operator` (the bytecode-level instruction enum).
//! - **Syscall fuel parameterization** (constant / linear / quadratic charging).
//! - **Hard upper-bounds** for syscall parameter sizes used in gas formulas, chosen so that
//!   the computations are safe under WebAssembly's 32-bit arithmetic constraints.
//!
//! ## Design goals (why these limits exist)
//! Fuel metering is frequently computed *inside the guest* (or inside instrumentation code
//! that is constrained by Wasm semantics). That has two practical consequences:
//!
//! 1. **Intermediate arithmetic must not overflow `i32`**
//!    Many metering implementations use `i32` ops (because Wasm has `i32` cheaply available,
//!    and because some runtimes lower instrumentation to `i32`). Overflow in Wasm integer ops
//!    *wraps* rather than trapping, which is a classic foot-gun: a wrap can turn a huge cost
//!    into a small one.
//!
//! 2. **Final fuel must fit into the chosen host type**
//!    In this crate we expose most fuel parameters as `u64` for flexibility, but the *per
//!    instruction* accounting often ends up in `u32` or `u64` depending on the embedding.
//!
//! Because of (1), this file defines conservative `*_MAX_*` constants that bound the input
//! parameter `x` for linear/quadratic formulas.
//!
//! ## `no_std` note
//! This crate is `no_std`-friendly via `cfg_attr`, but some types below (e.g. `String`) require
//! `alloc`. If you build without the `std` feature, ensure you also provide `alloc` in the
//! dependency graph (typically by enabling an `alloc` feature in your crate root).
//!
//! The exact split is project-specific, so we keep the public API unchanged here.

extern crate alloc;

use alloc::string::String;
use core::num::NonZeroU32;

/// Upper bound for the input parameter `x` used by **linear** syscall/builtin gas formulas.
///
/// Many builtins charge fuel proportional to the number of 32-byte "words" touched by an input:
///
/// ```text
/// words = (x + 31) / 32
/// fuel  = base_cost + word_cost * words
/// ```
///
/// ### Why do we cap `x`?
/// The cap is chosen so that *all intermediate arithmetic* can be implemented safely with
/// `i32` operations in Wasm instrumentation without overflow/wrap-around.
///
/// The tightest intermediate is typically the multiplication:
///
/// ```text
/// words * word_cost <= i32::MAX
/// ```
///
/// Approximating `words ≈ x / 32`, we obtain:
///
/// ```text
/// (x / 32) * word_cost_max <= 2^31 - 1
/// x <= (2^31 * 32) / word_cost_max
/// ```
///
/// ### Worst-case parameters
/// In this codebase, the worst (largest) `word_cost` comes from the most expensive linear
/// builtin (historically `DEBUG_LOG` when `FUEL_DENOM_RATE = 20`), giving:
///
/// - `word_cost_max = 16 * 20 = 320`
///
/// which yields a theoretical upper bound of ~204 MiB.
///
/// We intentionally choose a smaller, round power-of-two-ish bound (128 MiB) as a safety margin.
pub const FUEL_MAX_LINEAR_X: u32 = 134_217_728; // 128 MiB (2^27)

/// Upper bound for the input parameter `x` used by **quadratic** syscall/builtin gas formulas.
///
/// Some operations (e.g. those with potential quadratic behavior) use a formula like:
///
/// ```text
/// words = (x + 31) / 32
/// fuel  = (word_cost * words + words^2 / divisor) * FUEL_DENOM_RATE
/// ```
///
/// ### Why do we cap `x`?
/// In the quadratic case, the dangerous intermediate is `words * words`. We cap `x` such that
/// `words^2` fits in `i32`:
///
/// ```text
/// words^2 <= i32::MAX
/// words   <= floor(sqrt(i32::MAX)) = 46340
/// x       <= 46340 * 32 = 1_482_880 bytes (~1.4 MiB)
/// ```
///
/// We again choose a smaller bound (≈1.25 MiB) for margin.
pub const FUEL_MAX_QUADRATIC_X: u32 = 1_310_720; // 1.25 MiB (2^20 + 2^18)

/// Fuel charging policy for a syscall.
///
/// Syscalls may be charged in different ways depending on their asymptotic cost:
/// - **None**: no fuel is charged (used for free / accounting-only syscalls).
/// - **Const**: fixed cost, regardless of parameters.
/// - **LinearFuel**: cost scales linearly with a chosen parameter (e.g. length in bytes).
/// - **QuadraticFuel**: cost has a quadratic component (e.g. operations on dynamic arrays).
///
/// The actual charging logic typically lives in the runtime/instrumentation layer; this enum
/// is just the configuration payload.
#[derive(Debug, Clone, PartialEq)]
pub enum SyscallFuelParams {
    /// Fuel is not charged.
    None,

    /// Fuel is charged as a constant value.
    Const(ConstantFuelParams),

    /// Fuel is charged according to a linear formula, based on one of the syscall parameters.
    LinearFuel(LinearFuelParams),

    /// Fuel is charged according to a quadratic formula, based on one of the syscall parameters.
    QuadraticFuel(QuadraticFuelParams),
}

impl Default for SyscallFuelParams {
    fn default() -> Self {
        Self::None
    }
}

/// Constant fuel parameter type.
///
/// We use `u64` to avoid accidental truncation when combining multiple components.
pub type ConstantFuelParams = u64;

/// Parameters for the **linear** fuel charging strategy.
///
/// The common pattern is:
///
/// ```text
/// x     = syscall_args[linear_param_index]
/// words = (x + 31) / 32
/// fuel  = base_fuel + word_cost * words
/// ```
///
/// `max_linear` is a guardrail: it is typically set to [`FUEL_MAX_LINEAR_X`] or a smaller
/// syscall-specific limit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinearFuelParams {
    /// Base fuel charged for the syscall, independent of `x`.
    pub base_fuel: u32,

    /// Which syscall argument is treated as the size-like parameter `x`.
    ///
    /// This is expressed as an index into the syscall's parameter list as seen by the
    /// instrumentation/runtime layer.
    pub linear_param_index: u32,

    /// Fuel per 32-byte word (after rounding `x` up to words).
    pub word_cost: u32,
}

/// Parameters for the **quadratic** fuel charging strategy.
///
/// A typical formula looks like:
///
/// ```text
/// x     = syscall_args[...]
/// words = (x + 31) / 32
/// fuel  = (word_cost * words + words^2 / divisor) * fuel_denom_rate
/// ```
///
/// In practice, the runtime might compute `x` from a more complex structure than a single
/// argument. `local_depth` exists to support those calling conventions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuadraticFuelParams {
    /// Depth or "level" in the local parameter structure where the size-like value `x` lives.
    ///
    /// The exact meaning depends on the syscall ABI used by your runtime (e.g. nested tuples,
    /// indirect references, etc.). Keeping it explicit avoids baking ABI assumptions into the enum.
    pub local_depth: u32,

    /// Fuel per 32-byte word for the linear component.
    pub word_cost: u32,

    /// Divisor for the quadratic term `words^2 / divisor`.
    ///
    /// Larger divisors make the quadratic penalty gentler.
    pub divisor: u32,

    /// A multiplier used to scale fuel into a "denominated" unit.
    ///
    /// This is commonly used when you want finer granularity (e.g. to avoid fractional fuel)
    /// but still want integer arithmetic in the metering layer.
    pub fuel_denom_rate: u32,
}

/// Fully-qualified syscall name (module + function).
///
/// This is a lightweight key type, typically used in hash maps to look up fuel parameters or
/// dispatch implementations.
///
/// Note: this struct intentionally keeps `String` fields to simplify use in higher-level
/// configuration formats (JSON/YAML/etc.). If you need an allocation-free variant, consider
/// adding an interned or `&'static str` representation at the call site.
#[derive(Eq, Hash, PartialEq, Clone)]
pub struct SyscallName {
    /// Syscall module name (e.g. `"env"`).
    pub module: String,

    /// Syscall function name within the module (e.g. `"keccak256"`).
    pub name: String,
}

// -------------------------------------------------------------------------------------------------
// Per-operator fuel costs
// -------------------------------------------------------------------------------------------------
//
// These constants are intentionally small and expressed as `u32` because they are used in tight
// interpreter loops and in instrumentation logic.
//
// If you change these values, be mindful that:
// - Any ahead-of-time instrumentation may need to be re-generated.
// - Consensus-critical systems must ensure all nodes use the same schedule.
//

/// Default cost for "ordinary" operators that don't fall into other buckets.
pub const BASE_FUEL_COST: u32 = 1;

/// Cost for operators that touch global/module entities (tables, memories, globals, etc.).
pub const ENTITY_FUEL_COST: u32 = 3;

/// Cost for load instructions.
pub const LOAD_FUEL_COST: u32 = 2;

/// Cost for store instructions.
pub const STORE_FUEL_COST: u32 = 2;

/// Cost for direct and indirect calls.
///
/// Calls are expensive because they can:
/// - Change control flow non-locally,
/// - Introduce call-frame management,
/// - Trigger host<->guest transitions (for imported functions).
pub const CALL_FUEL_COST: u32 = 10;

// -------------------------------------------------------------------------------------------------
// Metering granularity constants
// -------------------------------------------------------------------------------------------------
//
// These define how many "units" correspond to one unit of fuel in some cost models.
// They are paired with *_LOG2 values so implementations can use shifts instead of division.
//

/// Number of memory bytes that correspond to one unit of fuel (must be a power of two).
pub const MEMORY_BYTES_PER_FUEL: u32 = 64;
/// `log2(MEMORY_BYTES_PER_FUEL)`.
pub const MEMORY_BYTES_PER_FUEL_LOG2: u32 = 6;

/// Number of table elements that correspond to one unit of fuel (must be a power of two).
pub const TABLE_ELEMS_PER_FUEL: u32 = 16;
/// `log2(TABLE_ELEMS_PER_FUEL)`.
pub const TABLE_ELEMS_PER_FUEL_LOG2: u32 = 4;

/// Number of locals that correspond to one unit of fuel (currently 1).
pub const LOCALS_PER_FUEL: u32 = 1;
/// `log2(LOCALS_PER_FUEL)` (0 because LOCALS_PER_FUEL == 1).
pub const LOCALS_PER_FUEL_LOG2: u32 = 0;

/// Drop/keep stack manipulation granularity (must be a power of two).
pub const DROP_KEEP_PER_FUEL: u32 = 16;
/// `log2(DROP_KEEP_PER_FUEL)`.
pub const DROP_KEEP_PER_FUEL_LOG2: u32 = 4;

// Compile-time invariants: the *_LOG2 constants must match the power-of-two values above.
//
// This keeps the shift-based implementations honest: if someone edits the constants and forgets
// to update the log2 value, compilation fails immediately (fast feedback, fewer "silent" bugs).
const _: () = assert!(MEMORY_BYTES_PER_FUEL == (1 << MEMORY_BYTES_PER_FUEL_LOG2));
const _: () = assert!(TABLE_ELEMS_PER_FUEL == (1 << TABLE_ELEMS_PER_FUEL_LOG2));
const _: () = assert!(LOCALS_PER_FUEL == (1 << LOCALS_PER_FUEL_LOG2));
const _: () = assert!(DROP_KEEP_PER_FUEL == (1 << DROP_KEEP_PER_FUEL_LOG2));

/// Returns the fuel cost for a single WebAssembly operator.
///
/// This function implements a **coarse, schedule-based** cost model:
/// - Many simple ops cost [`BASE_FUEL_COST`].
/// - Memory loads/stores use their own buckets.
/// - Calls are deliberately more expensive.
/// - Some "zero-cost" ops are treated as free because they typically compile to nothing
///   (e.g. `nop`, `drop`) or are structural markers (`block`, `loop`, `end`, ...).
///
/// ### Important subtlety
/// This is *not* intended to be a perfect cycle-accurate model. It's a deterministic "metering
/// schedule" whose primary job is to:
/// - bound execution,
/// - make denial-of-service expensive,
/// - be stable enough for reproducible execution across nodes.
///
/// If you tighten or relax costs here, double-check any consensus assumptions and test vectors.
pub fn rwasm_fuel_for_operator(op: &wasmparser::Operator) -> u32 {
    use wasmparser::Operator::*;

    match op {
        // `nop` and `drop` usually generate no machine code (or are optimized away), so they
        // don't consume fuel in this schedule.
        Nop | Drop => 0,

        // Structural control flow operators are "free" here because their runtime cost is
        // accounted for by the actual executed operators within their bodies.
        //
        // Note: we intentionally do NOT include `If` here; the conditional check itself has a
        // cost in most engines.
        Block { .. } | Loop { .. } | Unreachable | Return | Else | End => 0,

        // All forms of calls share the same bucket in this schedule.
        Call { .. } | CallIndirect { .. } | ReturnCall { .. } | ReturnCallIndirect { .. } => {
            CALL_FUEL_COST
        }

        // Loads.
        I32Load { .. }
        | I64Load { .. }
        | F32Load { .. }
        | F64Load { .. }
        | I32Load8S { .. }
        | I32Load8U { .. }
        | I32Load16S { .. }
        | I32Load16U { .. }
        | I64Load8S { .. }
        | I64Load8U { .. }
        | I64Load16S { .. }
        | I64Load16U { .. }
        | I64Load32S { .. }
        | I64Load32U { .. } => LOAD_FUEL_COST,

        // Stores.
        I32Store { .. }
        | I64Store { .. }
        | F32Store { .. }
        | F64Store { .. }
        | I32Store8 { .. }
        | I32Store16 { .. }
        | I64Store8 { .. }
        | I64Store16 { .. }
        | I64Store32 { .. } => STORE_FUEL_COST,

        // Entity operations that touch global state or metadata.
        GlobalGet { .. }
        | GlobalSet { .. }
        | MemorySize { .. }
        | MemoryGrow { .. }
        | MemoryInit { .. }
        | DataDrop { .. }
        | MemoryCopy { .. }
        | MemoryFill { .. }
        | TableInit { .. }
        | ElemDrop { .. }
        | TableCopy { .. }
        | TableFill { .. }
        | TableGet { .. }
        | TableSet { .. }
        | TableGrow { .. }
        | TableSize { .. } => ENTITY_FUEL_COST,

        // Everything else gets the base cost.
        _ => BASE_FUEL_COST,
    }
}

/// Type storing all kinds of fuel costs of instructions.
#[derive(Default, Debug, Copy, Clone)]
pub struct FuelCosts;

impl FuelCosts {
    pub const BASE: u32 = BASE_FUEL_COST;
    pub const ENTITY: u32 = ENTITY_FUEL_COST;
    pub const LOAD: u32 = LOAD_FUEL_COST;
    pub const STORE: u32 = STORE_FUEL_COST;
    pub const CALL: u32 = CALL_FUEL_COST;

    /// Returns the fuel consumption of the number of items with costs per items.
    pub fn costs_per(len_items: u32, items_per_fuel: u32) -> u32 {
        if len_items == 0 {
            return 0;
        }
        NonZeroU32::new(items_per_fuel)
            .map(|items_per_fuel_nz| {
                (len_items.saturating_add(items_per_fuel) - 1) / items_per_fuel_nz
            })
            .unwrap_or(0)
    }

    /// Returns the fuel consumption for branches and returns using the given [`DropKeep`].
    pub fn fuel_for_drop_keep(drop: u16, keep: u16) -> u32 {
        if drop == 0 {
            return 0;
        }
        Self::costs_per(u32::from(keep), DROP_KEEP_PER_FUEL)
    }

    /// Returns the fuel consumption for calling a function with the amount of local variables.
    ///
    /// # Note
    ///
    /// Function parameters are also treated as local variables.
    pub fn fuel_for_locals(locals: u32) -> u32 {
        Self::costs_per(locals, LOCALS_PER_FUEL)
    }

    /// Returns the fuel consumption for processing the amount of memory bytes.
    pub fn fuel_for_bytes(bytes: u32) -> u32 {
        Self::costs_per(bytes, MEMORY_BYTES_PER_FUEL)
    }

    /// Returns the fuel consumption for processing the amount of table elements.
    pub fn fuel_for_elements(elements: u32) -> u32 {
        Self::costs_per(elements, TABLE_ELEMS_PER_FUEL)
    }
}

/// rWasm disable several opcodes, most of them are FPU related
pub fn is_rwasm_operator_disabled(op: &wasmparser::Operator) -> bool {
    use wasmparser::Operator::*;
    match op {
        F32Load { .. }
        | F64Load { .. }
        | F32Store { .. }
        | F64Store { .. }
        | F32Eq
        | F32Ne
        | F32Lt
        | F32Gt
        | F32Le
        | F32Ge
        | F64Eq
        | F64Ne
        | F64Lt
        | F64Gt
        | F64Le
        | F64Ge
        | F32Abs
        | F32Neg
        | F32Ceil
        | F32Floor
        | F32Trunc
        | F32Nearest
        | F32Sqrt
        | F32Add
        | F32Sub
        | F32Mul
        | F32Div
        | F32Min
        | F32Max
        | F32Copysign
        | F64Abs
        | F64Neg
        | F64Ceil
        | F64Floor
        | F64Trunc
        | F64Nearest
        | F64Sqrt
        | F64Add
        | F64Sub
        | F64Mul
        | F64Div
        | F64Min
        | F64Max
        | F64Copysign
        | I32TruncF32S
        | I32TruncF32U
        | I32TruncF64S
        | I32TruncF64U
        | I64TruncF32S
        | I64TruncF32U
        | I64TruncF64S
        | I64TruncF64U
        | F32ConvertI32S
        | F32ConvertI32U
        | F32ConvertI64S
        | F32ConvertI64U
        | F32DemoteF64
        | F64ConvertI32S
        | F64ConvertI32U
        | F64ConvertI64S
        | F64ConvertI64U
        | F64PromoteF32
        | I32TruncSatF32S
        | I32TruncSatF32U
        | I32TruncSatF64S
        | I32TruncSatF64U
        | I64TruncSatF32S
        | I64TruncSatF32U
        | I64TruncSatF64S
        | I64TruncSatF64U => true,
        _ => false,
    }
}
