#![cfg_attr(not(feature = "std"), no_std)]
mod fuel_costs;
pub use fuel_costs::*;

/// The maximum allowed value for the `x` parameter used in linear gas cost calculation
/// of builtins.
/// This limit ensures:
/// 1. Runtime: all intermediate i32 operations do not overflow (WASM constraint)
/// 2. Compile-time: final fuel value fits in u32
///
/// Formula:
/// words = (x + 31) / 32
/// fuel = base_cost + word_cost × words
///
/// Derivation:
/// The bottleneck is I32Mul: words × word_cost ≤ i32::MAX
///
/// words ≈ x / 32, so:
/// (x / 32) × word_cost ≤ 2^31
/// x ≤ (2^31 × 32) / word_cost_max
///
/// Worst case (DEBUG_LOG with FUEL_DENOM_RATE = 20):
/// - word_cost_max = 16 × 20 = 320
///
/// x ≤ (2^31 × 32) / 320 = 214,748,364 bytes (~204 MB)
///
/// We use 128 MB as a safe limit within the theoretical maximum:
pub const FUEL_MAX_LINEAR_X: u32 = 134_217_728; // 128 MB (2^27)

/// The maximum allowed value for the `x` parameter used in quadratic gas cost calculation
/// of builtins.
/// This limit ensures:
/// 1. Runtime: words × words does not overflow i32 (WASM constraint)
/// 2. Compile-time: final fuel value fits in u32
///
/// Formula:
/// words = (x + 31) / 32
/// fuel = (word_cost × words + words² / divisor) × FUEL_DENOM_RATE
///
/// Derivation:
/// words × words must not overflow i32:
/// words² ≤ i32::MAX (2,147,483,647)
/// words ≤ 46,340
/// x ≤ 46,340 × 32 = 1,482,880 bytes (~1.4 MB)
///
/// We use 1.25 MB as a safe limit within the theoretical maximum:
pub const FUEL_MAX_QUADRATIC_X: u32 = 1_310_720; // 1.25 MB (2^20 + 2^18)

pub const BASE_FUEL_COST: u32 = 1;
pub const ENTITY_FUEL_COST: u32 = 3;
pub const LOAD_FUEL_COST: u32 = 2;
pub const STORE_FUEL_COST: u32 = 2;
pub const CALL_FUEL_COST: u32 = 10;
pub const STRUCT_FUEL_COST: u32 = 5;

pub const MEMORY_BYTES_PER_FUEL: u32 = 64;
pub const MEMORY_BYTES_PER_FUEL_LOG2: u32 = 6;
pub const TABLE_ELEMS_PER_FUEL: u32 = 16;
pub const TABLE_ELEMS_PER_FUEL_LOG2: u32 = 4;
pub const LOCALS_PER_FUEL: u32 = 1;
pub const LOCALS_PER_FUEL_LOG2: u32 = 0;
pub const DROP_KEEP_PER_FUEL: u32 = 16;
pub const DROP_KEEP_PER_FUEL_LOG2: u32 = 4;

const _: () = assert!(MEMORY_BYTES_PER_FUEL == (1 << MEMORY_BYTES_PER_FUEL_LOG2));
const _: () = assert!(TABLE_ELEMS_PER_FUEL == (1 << TABLE_ELEMS_PER_FUEL_LOG2));
const _: () = assert!(LOCALS_PER_FUEL == (1 << LOCALS_PER_FUEL_LOG2));
const _: () = assert!(DROP_KEEP_PER_FUEL == (1 << DROP_KEEP_PER_FUEL_LOG2));

pub fn fuel_for_operator(op: &wasmparser::Operator) -> u32 {
    use wasmparser::Operator::*;
    match op {
        Call { .. } | CallIndirect { .. } => CALL_FUEL_COST,
        GlobalGet { .. } | GlobalSet { .. } => ENTITY_FUEL_COST,
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
        I32Store { .. }
        | I64Store { .. }
        | F32Store { .. }
        | F64Store { .. }
        | I32Store8 { .. }
        | I32Store16 { .. }
        | I64Store8 { .. }
        | I64Store16 { .. }
        | I64Store32 { .. } => STORE_FUEL_COST,
        MemorySize { .. }
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
        ReturnCall { .. } => CALL_FUEL_COST,
        ReturnCallIndirect { .. } => CALL_FUEL_COST,
        _ => BASE_FUEL_COST,
    }
}
