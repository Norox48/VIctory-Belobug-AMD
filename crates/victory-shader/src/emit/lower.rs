//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

mod methods;

use std::collections::{BTreeMap, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Rmw, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use super::{
    OWN_SLOTS, VIEW_SLOTS, declares_structured_buffer, limits_indices, loop_limit, rays, shader,
    wave,
};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// From the vertex values of a constant element: P0 when `p0`, plus
    /// i (P1 - P0), plus j (P2 - P0), weights as floats.
    Weights {
        p0: bool,
        i: Option<V>,
        j: Option<V>,
    },
}

/// Updates which have no native DXIL atomic instruction.
#[derive(Clone, Copy)]
enum AtomicUpdate {
    Inc,
    Dec,
    ReverseSub,
    MaskOr,
    Wrap,
    ClampSub,
}

/// An unformatted byte or halfword load. LLVM's DSInstructions.td and
/// BUFInstructions.td model the D16 destinations as tied to their old value.
fn subword_load(rest: &str) -> Option<(u32, bool, Option<u32>)> {
    let (rest, half) = if let Some(n) = rest.strip_suffix("_d16_hi") {
        (n, Some(16))
    } else if let Some(n) = rest.strip_suffix("_d16") {
        (n, Some(0))
    } else {
        (rest, None)
    };
    let (width, signed) = match rest {
        "i8" | "sbyte" => (8, true),
        "u8" | "ubyte" => (8, false),
        "i16" | "sshort" => (16, true),
        "u16" | "ushort" | "short" => (16, false),
        _ => return None,
    };
    Some((width, signed, half))
}

fn ds_integer_atomic(name: &str) -> Option<(&str, bool, u8)> {
    let (op, ty) = name.strip_prefix("ds_")?.rsplit_once('_')?;
    let op = op.strip_suffix("_rtn").unwrap_or(op);
    let wide = match ty {
        "b32" | "u32" | "i32" => false,
        "b64" | "u64" | "i64" => true,
        _ => return None,
    };
    match op {
        "add" | "sub" | "and" | "or" | "xor" | "min" | "max" | "wrxchg" | "wrxchg2"
        | "wrxchg2st64" | "rsub" | "inc" | "dec" | "mskor" | "wrap" | "cmpst" => {
            Some((op, wide, ty.as_bytes()[0]))
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    pub stride_table: bool,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Test builds: the property words of each handle annotation, in order.
    #[cfg(test)]
    pub annotations: Vec<(u32, u32)>,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub psv: Option<&'a [u8]>,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// The wave sizes the host runs.
    pub target: super::Target,
    /// Run the program as the console does (see [`CONSOLE_FLOW`]); `st`
    /// is then [`crate::structure::structure_console`]'s.
    pub console_cf: bool,
    /// Never ask for the console flow: the per-lane form or nothing (what
    /// a program falls back to when the console flow cannot lower it).
    pub per_lane_only: bool,
}

#[cfg(test)]
thread_local! {
    /// The opcode audit: the instruction (by code offset) whose refusal is
    /// reported even when none of its results are used.
    pub(crate) static AUDIT_OFFSET: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// What [`lower`] stops with when a program should run as the console runs
/// it: lower it again from [`crate::structure::structure_console`] with
/// [`Parts::console_cf`]. Every lane of a console wave then runs every
/// block, branches are the console's (taken wave-wide) and EXEC guards the
/// vector instructions. Two kinds of program ask for it:
///
/// * a wave64 program whose two 32-lane halves must exchange values on a
///   host without 64-lane waves (a group of one console wave): the wave-wide
///   tests come from masks both halves exchange through group-shared
///   memory, and both halves reach every exchange, so a group barrier
///   orders it;
/// * on any host, a program that reads or writes a named lane
///   (`v_readlane`, `v_writelane`) and has branches its lanes take
///   differently. Those instructions ignore EXEC on the console, and a wave
///   runs a loop while any lane does; in the per-lane form a lane that left
///   a loop (or skipped a branch) is not there to be read, and never sees
///   the scalars set inside. One host wave is the console wave here, so
///   nothing is exchanged. If the console flow cannot lower such a program
///   it keeps the per-lane form ([`Parts::per_lane_only`]).
pub(crate) const CONSOLE_FLOW: &str = "a wave64 program whose halves exchange lanes (console flow)";

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

const UNTRACKED_SELECT: Why = "a select on an untracked mask";

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    match lower_program(parts, false) {
        Err(ShaderError::Unsupported {
            what: UNTRACKED_SELECT,
            ..
        }) if parts.native.stage == HwStage::Cs => lower_program(parts, true),
        result => result,
    }
}

fn lower_program(parts: &Parts, numeric_selects: bool) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    loop {
        match Lower::new(parts, &bad, numeric_selects).run() {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

/// The f16 bits of inline float constant `c` (240..=248: 0.5, -0.5, 1.0,
/// -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2 pi)).
fn inline_f16(c: u8) -> u32 {
    [
        0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
    ][usize::from(c - 240)]
}

/// Private memory per lane when the code does not show how far its
/// addresses reach (an index loaded or carried around a loop).
// shortcut: the console's size is set at pipeline creation (the program
// metadata read so far does not hold it); accesses past this read 0 and
// drop writes, as accesses past the console's size do.
pub(super) const SCRATCH_CAP: u32 = 4 << 10;
/// The most private memory per lane, the D3D11 limit of indexable
/// temporaries (4096 four-dword registers).
// shortcut: accesses past it read 0 and drop writes.
const SCRATCH_MAX: u32 = 64 << 10;

/// Bytes a `scratch_*` opcode moves (per lane), from the name after
/// `scratch_load_` or `scratch_store_`.
fn scratch_width(rest: &str) -> Option<u32> {
    Some(match rest.split('_').next()? {
        "dword" => 4,
        "dwordx2" => 8,
        "dwordx3" => 12,
        "dwordx4" => 16,
        "byte" | "ubyte" | "sbyte" => 1,
        "short" | "ushort" | "sshort" => 2,
        _ => return None,
    })
}

/// Private memory per lane the program needs: past the highest byte its
/// scratch accesses can reach where the code computing their addresses
/// bounds them, else [`SCRATCH_CAP`]; at most [`SCRATCH_MAX`].
fn scratch_bytes(p: &Program) -> u32 {
    let mut defs = HashMap::new();
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        for &(_, v) in &i.writes {
            defs.insert(v, i);
        }
    }
    let mut memo = HashMap::new();
    let mut end = 0u64;
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        let name = i.name();
        let Fields::Flat {
            addr,
            saddr,
            offset,
            ..
        } = &i.fields
        else {
            continue;
        };
        let rest = name
            .strip_prefix("scratch_load_")
            .or_else(|| name.strip_prefix("scratch_store_"));
        let Some(width) = rest.and_then(scratch_width) else {
            continue;
        };
        let base = match (addr, saddr) {
            (Some(o), _) | (None, Some(o)) => operand_bound(&defs, &mut memo, i, o, 0),
            (None, None) => Some(0),
        };
        // A negative offset only lowers the address (or wraps it out of
        // range, checked when the program runs).
        let off = u64::from((*offset).max(0).unsigned_abs());
        end = end.max(base.map_or(u64::from(SCRATCH_CAP), |b| b + off + u64::from(width)));
    }
    end.min(u64::from(SCRATCH_MAX)) as u32
}

/// An upper bound of operand `o` of `inst` as an unsigned number.
fn operand_bound(
    defs: &HashMap<Value, &Inst>,
    memo: &mut HashMap<Value, Option<u64>>,
    inst: &Inst,
    o: &Operand,
    depth: u8,
) -> Option<u64> {
    match *o {
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Scalar { code, .. } => value_bound(defs, memo, inst.read(Reg::S(code))?, depth),
        Operand::Vector { reg, .. } => value_bound(defs, memo, inst.read(Reg::V(reg))?, depth),
        Operand::Const(c) => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        Operand::Attr { .. } => None,
    }
}

/// An upper bound of 32-bit value `v` as an unsigned number, where the
/// instructions computing it show one (constants combined by masks, shifts,
/// sums, products, selects); `None` where it depends on anything else
/// (inputs, memory, a phi) or may wrap.
fn value_bound(
    defs: &HashMap<Value, &Inst>,
    memo: &mut HashMap<Value, Option<u64>>,
    v: Value,
    depth: u8,
) -> Option<u64> {
    if let Some(&b) = memo.get(&v) {
        return b;
    }
    let inst = *defs.get(&v)?;
    if depth > 32 {
        return None;
    }
    let (dst, src) = match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            imm: None,
        } => (Reg::S(*code), src),
        Fields::Valu(x)
            if x.sdwa.is_none()
                && x.dpp.is_none()
                && x.neg | x.abs | x.opsel == 0
                && !x.clamp
                && x.omod == 0 =>
        {
            match x.dst {
                Some(Operand::Vector { reg, .. }) => (Reg::V(reg), &x.src),
                _ => return None,
            }
        }
        _ => return None,
    };
    if inst.write(dst) != Some(v) {
        return None;
    }
    let mut arg =
        |k: usize| -> Option<u64> { operand_bound(defs, memo, inst, src.get(k)?, depth + 1) };
    // The smallest all-ones number at least `x` (bounds `|` and `^`).
    let ones = |x: u64| {
        if x == 0 {
            0
        } else {
            u64::MAX >> x.leading_zeros()
        }
    };
    let u24 = |x: Option<u64>| x.map(|x| x.min(0xFF_FFFF));
    let shl = |x: Option<u64>, n: Option<u64>| -> Option<u64> {
        x?.checked_shl(u32::try_from(n.filter(|&n| n < 32)?).ok()?)
    };
    let b = match inst.name() {
        "v_mov_b32" | "s_mov_b32" => arg(0),
        "v_and_b32" | "s_and_b32" => match (arg(0), arg(1)) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        },
        "v_or_b32" | "s_or_b32" | "v_xor_b32" | "s_xor_b32" => Some(ones(arg(0)?.max(arg(1)?))),
        "v_lshlrev_b32" => {
            let n = arg(0);
            shl(arg(1), n)
        }
        "s_lshl_b32" => {
            let x = arg(0);
            shl(x, arg(1))
        }
        "v_lshrrev_b32" => arg(1),
        "s_lshr_b32" => arg(0),
        "v_add_nc_u32" | "v_add_u32" | "v_add_co_u32" | "s_add_u32" | "s_add_i32" => {
            Some(arg(0)? + arg(1)?)
        }
        "v_add3_u32" => Some(arg(0)? + arg(1)? + arg(2)?),
        "v_mul_u32_u24" => Some(u24(arg(0))? * u24(arg(1))?),
        "v_mul_lo_u32" | "s_mul_i32" => arg(0)?.checked_mul(arg(1)?),
        "v_mad_u32_u24" => Some(u24(arg(0))? * u24(arg(1))? + arg(2)?),
        "v_lshl_add_u32" | "v_lshl_or_b32" => {
            let x = arg(0);
            let n = arg(1);
            Some(shl(x, n)? + arg(2)?)
        }
        "v_add_lshl_u32" => {
            let x = Some(arg(0)? + arg(1)?);
            shl(x, arg(2))
        }
        "v_and_or_b32" => {
            let x = match (arg(0), arg(1)) {
                (Some(x), Some(y)) => x.min(y),
                (x, y) => x.or(y)?,
            };
            Some(x + arg(2)?)
        }
        "v_bfe_u32" => match arg(2) {
            Some(w) if w < 32 => Some((1 << w) - 1),
            _ => arg(0),
        },
        "v_cndmask_b32" | "v_max_u32" | "s_max_u32" | "s_cselect_b32" => Some(arg(0)?.max(arg(1)?)),
        "v_min_u32" | "s_min_u32" => match (arg(0), arg(1)) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        },
        _ => None,
    }
    .filter(|&b| b <= u64::from(u32::MAX));
    memo.insert(v, b);
    b
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    /// SGPR spills to VGPR lanes and their reloads that nothing reads any
    /// more (see [`crate::sgpr_spill`]): not lowered.
    dead: HashSet<Site>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, (u32, u32)), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    wide_lds: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Retry a compute shader with numeric selects only when the normal
    /// translation needs one. Unused selects need no fixed wave size.
    numeric_selects: bool,
    /// A console wave64 compute program run as two 32-lane host waves
    /// (each lane's reads stay in its half; see `Lower::new`).
    split: bool,
    /// Split, and run as the console runs it ([`CONSOLE_FLOW`]).
    console_cf: bool,
    /// Split without [`Self::console_cf`] cannot run it, the console flow can.
    wants_console_flow: bool,
    /// The console thread's flat id in the group (relabelled; see
    /// [`Self::thread_id`]).
    console_flat: Option<V>,
    /// Group-shared slots the two halves exchange a dword through.
    xchg: Option<V>,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// Pixel inputs holding 16-bit values: row, whether two per dword, and
    /// the row packed after it.
    half_inputs: HashMap<u8, (bool, Option<u8>)>,
    /// Multi-element 16-bit rows whose trailing element overflows a 4-dword
    /// `pack2` attribute and spills into a free dword of a later `pack2`
    /// attribute: `(attr_row, half_index) -> (sig_row, col)`.
    half_input_spills: HashMap<(u8, u8), (u8, u8)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// SCC of a scalar mask operation ("some lane is set"), with the low
    /// dword of the mask it tests: read as data, the ballot is not zero.
    mask_scc: HashMap<Value, Value>,
    /// Lane masks read as numbers (see [`mask_numbers`]).
    mask_numbers: HashSet<Value>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// Values of whole-wave sections holding one lane's, the one read after.
    one_lane: HashSet<Value>,
    /// Memory by address is written too, for other groups to read.
    coherent: Option<bool>,
    /// Threads of different groups take turns through atomics (a tree built
    /// from its leaves up): what they wrote before one must be seen after
    /// it. The console's program waits for its writes there.
    ordered: Option<bool>,
    /// Views of buffers accessed by address, by instruction.
    made: HashMap<Site, V>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Private (scratch) memory: the lane's array and its dwords.
    scratch: Option<(V, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// Loops that lanes leave through one block, by that block: the header
    /// and the scalar values the loop writes that code after it reads (see
    /// [`Lower::leave_loops`]).
    loop_exits: HashMap<usize, Vec<(usize, Vec<Value>)>>,
    /// Loop bodies (IR blocks) by header.
    loop_bodies: HashMap<usize, HashSet<usize>>,
    /// Scalar values as the wave's last iteration of a divergent loop left
    /// them, with the loop's header: what code after the loop reads.
    finals: HashMap<Value, (Val, usize)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// Per-lane descriptor joins being lowered: IR block and the heap index
    /// phi there (what [`Source::Carried`] reads).
    lane_paths: Vec<(usize, V)>,
    /// Heap index phis of per-lane joins with predecessors lowered later.
    pending_paths: Vec<PendingPath>,
}

/// A heap index phi of a [`Source::Paths`], filled once every predecessor is
/// lowered.
struct PendingPath {
    phi: V,
    block: usize,
    sampler: bool,
    /// [`Lower::lane_paths`] where it was made, itself included.
    outer: Vec<(usize, V)>,
    /// Per predecessor: the index when it was lowered already, else its
    /// source.
    incoming: Vec<(usize, Result<V, Source>)>,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Scalar set operations, which keep a lane mask a lane mask: their result
/// as a number is the same operation on their sources' numbers.
fn mask_set_op(name: &str) -> bool {
    let Some(op) = name
        .strip_prefix("s_")
        .and_then(|n| n.strip_suffix("_b32").or_else(|| n.strip_suffix("_b64")))
    else {
        return false;
    };
    matches!(
        op,
        "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not" | "mov"
    )
}

/// Samples whose level comes from derivatives across each quad of lanes.
fn implicit_derivatives(name: &str) -> bool {
    sample_form(name).is_some_and(|f| !f.gather && !f.c && !f.d && !f.l && !f.lz)
}

/// The lane masks whose value as a number the program reads: those an
/// instruction reads as data ([`reads_mask_as_data`]) or a join that is no
/// mask takes, and the masks the set operations and joins writing them
/// read in turn. Only these get a number where a set operation writes them
/// (the others would take ballots nothing reads).
fn mask_numbers(st: &Structured, p: &crate::ir::Program) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut out: HashSet<Value> = HashSet::new();
    for b in &p.blocks {
        for phi in b.phis.iter().filter(|phi| !is_mask(phi.value)) {
            out.extend(phi.args.iter().copied().filter(|&a| is_mask(a)));
        }
        for i in b.insts.iter().filter(|i| reads_mask_as_data(st, i)) {
            let src: &[Operand] = match &i.fields {
                Fields::Sop { src, .. } => src,
                Fields::Valu(v) => &v.src,
                _ => &[],
            };
            for op in src {
                if let Operand::Scalar { code, dwords } = *op
                    && code != NULL
                {
                    let read = (0..dwords).filter_map(|k| i.read(Reg::S(code + k)));
                    out.extend(read.filter(|&v| is_mask(v)));
                }
            }
        }
    }
    loop {
        let before = out.len();
        for b in &p.blocks {
            for phi in &b.phis {
                if out.contains(&phi.value) {
                    out.extend(phi.args.iter().copied().filter(|&a| is_mask(a)));
                }
            }
            for i in &b.insts {
                if matches!(i.fields, Fields::Sop { .. })
                    && mask_set_op(i.name())
                    && i.writes.iter().any(|w| out.contains(&w.1))
                {
                    let read = i.reads.iter().filter(|r| matches!(r.0, Reg::S(_)));
                    out.extend(read.map(|r| r.1).filter(|&v| is_mask(v)));
                }
            }
        }
        if out.len() == before {
            return out;
        }
    }
}

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// SCC of each scalar operation writing a tracked lane mask, with the
/// mask's low dword (EXEC's new value for the `saveexec` forms).
fn mask_sccs(st: &Structured, p: &crate::ir::Program) -> HashMap<Value, Value> {
    let mut out = HashMap::new();
    for inst in p.blocks.iter().flat_map(|b| &b.insts) {
        let Fields::Sop { dst, .. } = &inst.fields else {
            continue;
        };
        let Some(scc) = inst.write(Reg::Scc) else {
            continue;
        };
        let lo = if inst.name().contains("saveexec") {
            inst.write(Reg::S(EXEC))
        } else {
            match *dst {
                Some(Operand::Scalar { code, .. }) => inst.write(Reg::S(code)),
                _ => None,
            }
        };
        if let Some(lo) = lo.filter(|lo| st.masks.contains_key(lo)) {
            out.insert(scc, lo);
        }
    }
    out
}

/// Whether `inst` gives each lane what it would under any EXEC: scalar
/// code and loads that do not touch EXEC, and vector code that reads no
/// other lane (no DPP, permutes or lane reads and writes).
fn lane_local(inst: &Inst) -> bool {
    // EXEC as an operand, or EXECZ.
    let exec = |o: &Operand| {
        matches!(*o, Operand::Scalar { code, .. } if code >= EXEC) || *o == Operand::Const(252)
    };
    let name = inst.name();
    match &inst.fields {
        Fields::Sop { .. } => !inst
            .reads
            .iter()
            .chain(&inst.writes)
            .any(|&(r, _)| r == Reg::S(EXEC) || r == Reg::S(EXEC + 1)),
        Fields::Smem { .. } => name.starts_with("s_load_") || name.starts_with("s_buffer_load_"),
        Fields::Valu(v) => {
            v.dpp.is_none()
                && !name.starts_with("v_interp_")
                && !matches!(
                    name,
                    "v_readlane_b32"
                        | "v_readfirstlane_b32"
                        | "v_writelane_b32"
                        | "v_permlane16_b32"
                        | "v_permlanex16_b32"
                )
                && !v.src.iter().chain(&v.dst).chain(&v.carry).any(exec)
        }
        _ => false,
    }
}

/// Dwords a scalar load reads, by the suffix of its name after
/// `s_load_dword` or `s_buffer_load_dword`.
fn smem_dwords(suffix: &str) -> Option<u32> {
    match suffix {
        "" => Some(1),
        "x2" => Some(2),
        "x4" => Some(4),
        "x8" => Some(8),
        "x16" => Some(16),
        _ => None,
    }
}

/// A select that needs the actual scalar mask words and console lane index.
fn numeric_select(st: &Structured, inst: &Inst) -> bool {
    inst.name() == "v_cndmask_b32"
        && matches!(&inst.fields, Fields::Valu(v)
            if matches!(v.src.get(2), Some(Operand::Scalar { code, .. }) if *code != NULL))
        && inst
            .writes
            .first()
            .is_some_and(|&(_, v)| !st.select_masks.contains_key(&v))
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// What a sample or gather opcode adds to the plain form, by its name:
/// comparison (`_c`), explicit derivatives (`_d`, coarse `_cd`, 16-bit
/// `_g16`), a level (`_l`) or level 0 (`_lz`), a bias (`_b`), a level
/// clamp (`_cl`), a texel offset (`_o`).
#[derive(Debug, Clone, Copy, Default)]
struct SampleForm {
    gather: bool,
    c: bool,
    d: bool,
    g16: bool,
    l: bool,
    lz: bool,
    b: bool,
    cl: bool,
    o: bool,
}

fn sample_form(name: &str) -> Option<SampleForm> {
    let (gather, rest) = match name.strip_prefix("image_sample") {
        Some(rest) => (false, rest),
        None => (true, name.strip_prefix("image_gather4")?),
    };
    let mut f = SampleForm {
        gather,
        ..SampleForm::default()
    };
    for token in rest.split('_').skip(1) {
        let flag = match token {
            "c" => &mut f.c,
            "d" | "cd" => &mut f.d,
            "g16" => &mut f.g16,
            "l" => &mut f.l,
            "lz" => &mut f.lz,
            "b" => &mut f.b,
            "cl" => &mut f.cl,
            "o" => &mut f.o,
            _ => return None,
        };
        *flag = true;
    }
    Some(f)
}

/// Pixel inputs of 16-bit values (see [`Lower::half_inputs`]): a packed row's
/// partner is `row2` when distinct from `row`, else the next unlisted 16-bit
/// row of the signature before the next `fp16` input row, taken in row order.
fn half_inputs(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    // Keep the first fp16 entry per row (matching the previous `.find()`
    // behaviour) so duplicate rows neither overwrite earlier ones nor consume
    // an extra free partner row.
    let mut seen = HashSet::new();
    let mut halves: Vec<_> = entries
        .iter()
        .filter(|i| i.fp16 && seen.insert(i.row))
        .collect();
    halves.sort_by_key(|i| i.row);
    // In the console's PsInput table, `row2` repeats `row` when no distinct
    // partner row is named; exclude both listed rows and explicitly claimed
    // partners (`row2 != row`) from the fallback pool.
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| {
            !entries.iter().any(|i| i.row == r)
                && !halves
                    .iter()
                    .any(|i| i.pack2 && i.row2 != i.row && i.row2 == r)
        })
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut out = HashMap::with_capacity(halves.len());
    for (idx, i) in halves.iter().enumerate() {
        let next_fp16 = halves.get(idx + 1).map_or(u8::MAX, |n| n.row);
        // Only take a fallback partner before the next fp16 input row so an
        // earlier unpaired input cannot steal a later fp16 input's partner.
        let b = if !i.pack2 {
            None
        } else if i.row2 != i.row {
            Some(i.row2)
        } else {
            free.iter()
                .position(|&r| r > i.row && r < next_fp16)
                .map(|p| free.remove(p))
        };
        out.insert(i.row, (i.pack2, b));
    }
    out
}

/// Elements from multi-element 16-bit signature rows that overflow a 4-dword
/// `pack2` attribute (because each element starts on a whole dword) and spill
/// into free dwords of a subsequent `pack2` attribute.
fn half_input_spills(
    half_inputs: &HashMap<u8, (bool, Option<u8>)>,
    inputs: &[SigElement],
) -> HashMap<(u8, u8), (u8, u8)> {
    let row_elems = |r: u8| -> Vec<&SigElement> {
        let mut v: Vec<_> = inputs
            .iter()
            .filter(|e| e.half && e.row == Some(u32::from(r)))
            .collect();
        v.sort_by_key(|e| e.col);
        v
    };
    let extent = |r: u8| -> u8 {
        inputs
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
            .unwrap_or(0)
    };
    let mut rows: Vec<(u8, Option<u8>)> = half_inputs
        .iter()
        .filter_map(|(&a, &(pack2, b))| pack2.then_some((a, b)))
        .collect();
    rows.sort_by_key(|&(a, _)| a);
    let mut pending: Vec<(u8, u8, u8)> = Vec::new();
    let mut out = HashMap::new();
    for (a, b) in rows {
        let mut elems_a = row_elems(a);
        let dwords_b: u8 = b
            .map(|r| row_elems(r).iter().map(|e| e.cols.div_ceil(2)).sum())
            .unwrap_or(0);
        let mut dwords_a: u8 = elems_a.iter().map(|e| e.cols.div_ceil(2)).sum();
        while elems_a.len() > 1 && dwords_a + dwords_b > 4 {
            let spill = elems_a.pop().unwrap();
            dwords_a -= spill.cols.div_ceil(2);
            pending.push((a, spill.col, spill.cols));
        }
        let mut used = extent(a).div_ceil(2) + b.map_or(0, |r| extent(r).div_ceil(2));
        if used < 4 && !pending.is_empty() {
            let mut idx = 0;
            while idx < pending.len() && used < 4 {
                let (sr, sc, scols) = pending[idx];
                let need = scols.div_ceil(2);
                if used + need <= 4 {
                    for k in 0..scols {
                        out.insert((a, used * 2 + k), (sr, sc + k));
                    }
                    used += need;
                    pending.remove(idx);
                } else {
                    idx += 1;
                }
            }
        }
    }
    out
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// A byte or halfword LDS access: `ds_read_{u,i}{8,16}[_d16[_hi]]` and
/// `ds_write_b{8,16}[_d16_hi]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubDword {
    write: bool,
    bits: u32,
    /// Reads: sign-extended.
    signed: bool,
    /// Reads: the result is 16 bits wide and goes into one half of the VGPR,
    /// the other half kept (`Some(true)`: the high half). Writes: the data
    /// comes from the high half (`Some(true)`).
    d16: Option<bool>,
}

impl SubDword {
    fn of(name: &str) -> Option<Self> {
        let (write, rest) = match name.strip_prefix("ds_read_") {
            Some(r) => (false, r),
            None => (true, name.strip_prefix("ds_write_")?),
        };
        let (ty, d16) = match rest.split_once("_d16") {
            None => (rest, None),
            Some((t, "")) if !write => (t, Some(false)),
            Some((t, "_hi")) => (t, Some(true)),
            Some(_) => return None,
        };
        let (signed, bits) = match (write, ty) {
            (false, "u8") => (false, 8),
            (false, "i8") => (true, 8),
            (false, "u16") => (false, 16),
            (false, "i16") => (true, 16),
            (true, "b8") => (false, 8),
            (true, "b16") => (false, 16),
            _ => return None,
        };
        Some(Self {
            write,
            bits,
            signed,
            d16,
        })
    }
}

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
/// The host's limit on a mesh shader's vertex plus primitive outputs
/// (DXC's `kMaxMSOutputTotalBytes`).
const MAX_MESH_OUTPUT_BYTES: u32 = 32768;

/// A mesh shader's output size as the validator counts it: every element's
/// components times the maximum vertex or primitive count, each rounded up
/// to 32, four bytes each.
fn mesh_output_bytes(
    outputs: &[SigElement],
    prims: &[SigElement],
    mesh: super::shader::Mesh,
) -> u32 {
    let scalars = |els: &[SigElement], n: u32| -> u32 {
        els.iter().map(|e| u32::from(e.cols)).sum::<u32>() * n.next_multiple_of(32)
    };
    4 * (scalars(outputs, mesh.max_vertices) + scalars(prims, mesh.max_primitives))
}

/// The coordinates `calculateLOD` takes for a texture of DXIL resource
/// kind `kind` (DXC's `GetNumDimensionsForCalcLOD`): the size's dimensions,
/// without an array slice. The validator wants the rest undef ("out of
/// bound coord must be undef").
fn lod_dims(kind: u32) -> usize {
    match kind {
        1 | 6 => 1,
        2 | 7 => 2,
        _ => 3,
    }
}

/// Gives the elements of each packed row one interpolation mode, as DXIL
/// requires ("interpolation mode that differs from another element packed
/// into the same row"). Elements whose mode is fixed (pixel inputs that
/// are read, integers and `SV_IsFrontFace`, which are constant, and the
/// position) set the row's mode; the others, floats nobody reads (vertex
/// outputs: any float), take it. Rows stay where the console put them:
/// the host links stages by register. A pixel row read two ways is an
/// error rather than a shader the validator rejects; `per_vertex` makes
/// rows with a constant element constant beforehand.
pub(super) fn settle_row_modes(elements: &mut [SigElement], pixel: bool) -> Result<(), Why> {
    // Packed elements only: barycentrics, sample index and coverage are not.
    let packed = |e: &SigElement| e.row.is_some() && !matches!(e.kind, 12 | 14 | 15 | 28 | 30);
    let fixed = |e: &SigElement| e.comp != 9 || e.kind == 3 || (pixel && e.usage != 0);
    let mut rows: Vec<u32> = elements
        .iter()
        .filter(|e| packed(e))
        .filter_map(|e| e.row)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    for row in rows {
        let in_row = |e: &SigElement| packed(e) && e.row == Some(row);
        let mut modes: Vec<u8> = elements
            .iter()
            .filter(|e| in_row(e) && fixed(e))
            .map(|e| e.interp)
            .collect();
        modes.sort_unstable();
        modes.dedup();
        let mode = match modes[..] {
            [m] => m,
            // Nothing fixed: the first element's.
            [] => match elements.iter().find(|e| in_row(e)) {
                Some(e) => e.interp,
                None => continue,
            },
            _ if pixel => return Err("input elements of one row interpolated two ways"),
            // Vertex outputs: the host interpolates by the pixel inputs' modes.
            _ => continue,
        };
        for e in elements.iter_mut().filter(|e| in_row(e) && !fixed(e)) {
            e.interp = mode;
        }
    }
    Ok(())
}

/// A pixel program under conservative depth declares its position centroid
/// (`emit`); one that runs per sample (an `SV_SampleIndex` input or a
/// per-sample interpolation) declares it per sample instead.
pub(super) fn settle_position_mode(inputs: &mut [SigElement]) {
    let per_sample = inputs
        .iter()
        .any(|e| e.kind == 12 || matches!(e.interp, 6 | 7));
    for e in inputs.iter_mut().filter(|e| e.kind == 3 && e.interp == 5) {
        if per_sample {
            e.interp = 7;
        }
    }
}

fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::container::SigElement;
    use super::{Lowered, Numeric, Parts, SubDword, buffer_format, half_shift, inline_f16, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{ShaderError, io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Mesh output size as the validator counts it (53 programs were over
    /// the limit; they now fail as unsupported).
    #[test]
    fn mesh_output_size_counts_aligned_maxima() {
        let el = |cols| SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(0),
            col: 0,
            cols,
            usage: 0,
            half: false,
        };
        let mesh = |max_vertices, max_primitives| crate::emit::shader::Mesh {
            max_vertices,
            max_primitives,
            topology: 2,
        };
        // 4 + 4 columns of 64 vertices, 1 column of 126 primitives (128).
        let size = super::mesh_output_bytes(&[el(4), el(4)], &[el(1)], mesh(64, 126));
        assert_eq!(size, 4 * (8 * 64 + 128));
        // 32 columns of 256 vertices: exactly the limit; one more is over.
        let full: Vec<_> = (0..8).map(|_| el(4)).collect();
        assert_eq!(
            super::mesh_output_bytes(&full, &[], mesh(256, 1)),
            super::MAX_MESH_OUTPUT_BYTES
        );
        assert!(
            super::mesh_output_bytes(&full, &[el(1)], mesh(256, 1)) > super::MAX_MESH_OUTPUT_BYTES
        );
    }

    /// `calculateLOD` takes the size's dimensions only, as DXC's
    /// `GetNumDimensionsForCalcLOD` counts them: an array's slice
    /// coordinate is left undef.
    #[test]
    fn level_queries_leave_the_array_slice_undef() {
        // 1D, 2D, 3D, cube, 1D array, 2D array.
        let dims: Vec<usize> = [1, 2, 4, 5, 6, 7].map(super::lod_dims).to_vec();
        assert_eq!(dims, [1, 2, 3, 3, 1, 2]);
    }

    /// VOP3 `op dst, a, b` (GFX10 encoding); `flags` are the clamp, op_sel
    /// and abs bits of the first dword.
    fn vop3(op: u32, flags: u32, dst: u32, a: u32, b: u32) -> [u32; 2] {
        [0xD400_0000 | op << 16 | flags | dst, a | b << 9]
    }

    /// The DXIL of a pixel shader that exports `body`'s v2, computed from
    /// a per-lane value in v3.
    fn lower_ps(body: [u32; 2]) -> Result<Vec<u8>, ShaderError> {
        let words = [
            // v_mbcnt_lo_u32_b32 v3, -1, 0
            vop3(0x365, 0, 3, 193, 128)[0],
            vop3(0x365, 0, 3, 193, 128)[1],
            body[0],
            body[1],
            0xF800_180F, // exp mrt0, v2, v2, v2, v2 done vm
            0x0202_0202,
            0xBF81_0000, // s_endpgm
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, false, 256)?;
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let io = io::raise(&p, &native)?;
        let res = resource::track(&p, &native.layout, native.stage)?;
        let st = structure::structure(&p)?;
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let parts = Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![target],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        };
        lower(&parts).map(|l| l.dxil)
    }

    /// Lowers a compute program of constants: whether it translates.
    fn lowers(words: &[u32]) -> Result<(), crate::ShaderError> {
        lower_with(words, vec![], vec![]).map(|_| ())
    }

    /// [`lowers`] with the input registers `rsrc2` names.
    fn lowers_rsrc2(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        lower_io(words, vec![], vec![], false, Some(rsrc2)).map(|_| ())
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_io(words, layout, used, wave64, None)
    }

    /// [`lower_wave`] with the input registers `rsrc2` names (default: the
    /// layout's user SGPRs, at least two, and 512 bytes of LDS).
    fn lower_io(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
        rsrc2: Option<u32>,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: rsrc2.unwrap_or(1 << 15 | sgprs.max(2) << 1),
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage)?;
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: super::super::Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    /// The view a root descriptor at `user_data` is.
    fn root_view(kind: ResourceType, user_data: u8, slot: u8) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data,
            slot,
        }
    }

    #[test]
    fn scalar_code_inside_a_whole_wave_section_is_lowered() {
        // Assembled with llvm-mc (gfx1030, wave32): scalar code the
        // compiler scheduled into a whole-wave section, branched on after.
        let words = [
            0xBE84_0385, // s_mov_b32 s4, 5
            0xBE80_3DC1, // s_or_saveexec_b32 s0, -1
            0x8005_8104, // s_add_u32 s5, s4, 1
            0x7E02_0280, // v_mov_b32_e32 v1, 0
            0xBEFE_0300, // s_mov_b32 exec_lo, s0
            0xBF06_8605, // s_cmp_eq_u32 s5, 6
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE86_0380, // s_mov_b32 s6, 0
            S_ENDPGM,
        ];
        lowers(&words).unwrap();
    }

    #[test]
    fn constants_loaded_from_a_root_view() {
        // Two dwords at byte 0x10 of the memory of a root descriptor (in
        // s[0:1]) read with s_load and branched on: read through the root
        // CBV, or the root SRV when the root signature says it is one.
        let words = [
            0xF404_0080,
            0xFA00_0010, // s_load_dwordx2 s[2:3], s[0:1], 0x10
            0x8002_0302, // s_add_u32 s2, s2, s3
            0xBF06_8502, // s_cmp_eq_u32 s2, 5
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Descriptor, 0, 3)];
        let cbv = lower_with(&words, layout(), vec![]).unwrap();
        assert_eq!((cbv.root_cbvs, cbv.root_srvs), (vec![3], vec![]));
        let used = vec![root_view(ResourceType::Srv, 0, 3)];
        let srv = lower_with(&words, layout(), used).unwrap();
        assert_eq!((srv.root_cbvs, srv.root_srvs), (vec![], vec![3]));
        // A root UAV's memory has no read-only host view.
        let used = vec![root_view(ResourceType::Uav, 0, 3)];
        assert!(lower_with(&words, layout(), used).is_err());
    }

    #[test]
    fn a_dword_of_a_descriptor_another_load_reads_whole() {
        // A 2D image's descriptor (a table range at s0) loaded whole for
        // image_get_resinfo, and its dword 2 (width and height) loaded
        // again alone and branched on.
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], null
            0x7E04_0480, // v_mov_b32 v2, 0
            0xF038_0308,
            0x0001_0402, // image_get_resinfo v[4:5], v2, s[4:11] dim:SQ_RSRC_IMG_2D
            0xF400_0300,
            0xFA00_0008, // s_load_dword s12, s[0:1], 0x8
            0xBF06_850C, // s_cmp_eq_u32 s12, 5
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0)];
        lower_with(&words, layout(), vec![]).unwrap();
        // Dword 2 of the next descriptor: no load of it is used.
        let mut next = words;
        next[7] = 0xFA00_0028;
        let e = lower_with(&next, layout(), vec![]).err().unwrap();
        assert!(
            e.to_string().contains("a descriptor read only as data"),
            "{e}"
        );
    }

    #[test]
    fn a_descriptor_dword_loaded_through_a_rebuilt_table_pointer() {
        // A 2D image's descriptor (a table range at s0) loaded whole for an
        // image_store in an exec-masked tail, and its dword 2 (width and
        // height) loaded alone through another copy of the table pointer
        // {s0, 0} (a copy in other registers, or the high half set to 0
        // again after the registers held scratch), its height bits read.
        let layout = || vec![root(RootType::Table, 0, 0)];
        let copied = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF400_0300,
            0xFA00_0008, // s_load_dword s12, s[0:1], 0x8
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF40C_0101,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[2:3], 0x0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x938D_FF0C,
            0x000E_000E, // s_bfe_u32 s13, s12, 0xe000e
            0x7E0A_020D, // v_mov_b32 v5, s13
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc_lo, 4, v0
            0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
            0xBF88_0002, // s_cbranch_execz -> s_endpgm
            0xF020_1F08,
            0x0001_0100, // image_store v[1:4], v[0:1], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
            S_ENDPGM,
        ];
        lower_with(&copied, layout(), vec![]).unwrap();
        let rebuilt = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], 0x0
            0x7D88_0084, // v_cmp_gt_u32 vcc_lo, 4, v0
            0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
            0xBF88_0002, // s_cbranch_execz -> s_or_b32
            0xF020_1F08,
            0x0001_0100, // image_store v[1:4], v[0:1], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0x7E08_0500, // v_readfirstlane_b32 s4, v0
            0x7E02_0501, // v_readfirstlane_b32 s1, v1
            0x8704_FF04,
            0x0000_01FF, // s_and_b32 s4, s4, 0x1ff
            0x8F01_8901, // s_lshl_b32 s1, s1, 9
            0x8804_0104, // s_or_b32 s4, s4, s1
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF400_0300,
            0xFA00_0008, // s_load_dword s12, s[0:1], 0x8
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x938D_FF0C,
            0x000E_000E, // s_bfe_u32 s13, s12, 0xe000e
            0x8804_0D04, // s_or_b32 s4, s4, s13
            0x7E0A_0204, // v_mov_b32 v5, s4
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            S_ENDPGM,
        ];
        lower_with(&rebuilt, layout(), vec![]).unwrap();
        // Dword 2 of the next descriptor: no load of it is used.
        let mut next = copied;
        next[2] = 0xFA00_0028;
        let e = lower_with(&next, layout(), vec![]).err().unwrap();
        assert!(
            e.to_string().contains("a descriptor read only as data"),
            "{e}"
        );
    }

    #[test]
    fn descriptor_registers_reused_after_a_masked_image_store() {
        // An image_store in an exec-masked tail is the only use of its
        // descriptor load; after the join the descriptor's registers hold
        // scratch (v_readfirstlane, then bit packing), alone or as the low
        // half of a 64-bit shift whose high half is still a descriptor
        // dword.
        let layout = || vec![root(RootType::Table, 0, 0)];
        let head = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], 0x0
            0x7D88_0084, // v_cmp_gt_u32 vcc_lo, 4, v0
            0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
            0xBF88_0002, // s_cbranch_execz -> past the store
            0xF020_1F08,
            0x0001_0100, // image_store v[1:4], v[0:1], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
        ];
        let packed = [
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0x7E08_0500, // v_readfirstlane_b32 s4, v0
            0x7E0A_0501, // v_readfirstlane_b32 s5, v1
            0x8704_FF04,
            0x0000_01FF, // s_and_b32 s4, s4, 0x1ff
            0x8F05_8905, // s_lshl_b32 s5, s5, 9
            0x8804_0504, // s_or_b32 s4, s4, s5
            0x7E0A_0204, // v_mov_b32 v5, s4
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            S_ENDPGM,
        ];
        let wide = [
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0x7E08_0500, // v_readfirstlane_b32 s4, v0
            0x8F84_8904, // s_lshl_b64 s[4:5], s[4:5], 9
            0x7E0A_0204, // v_mov_b32 v5, s4
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            S_ENDPGM,
        ];
        lower_with(&[&head[..], &[S_ENDPGM]].concat(), layout(), vec![]).unwrap();
        lower_with(&[&head[..], &packed].concat(), layout(), vec![]).unwrap();
        lower_with(&[&head[..], &wide].concat(), layout(), vec![]).unwrap();
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        for words in [add, sub] {
            lowers_rsrc2(&words, 0).unwrap();
        }
    }

    /// A dword load by index through a descriptor built around the root
    /// descriptor in s[0:1] with dword 1 bits `hi` and dword 3 `w3`, stored
    /// to LDS so the load is needed.
    fn indexed_root_load(hi: u32, w3: u32) -> Result<Lowered, crate::ShaderError> {
        let words = [
            0x8805_FF01,
            hi,          // s_or_b32 s5, s1, hi
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE86_03C1, // s_mov_b32 s6, -1
            0xBE87_03FF,
            w3, // s_mov_b32 s7, w3
            0xE030_2000,
            0x8001_0100, // buffer_load_dword v1, v0, s[4:7], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let root = RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 3,
        };
        // The container lists the root SRV without a stride.
        let used = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 3,
        };
        lower_with(&words, vec![root], vec![used])
    }

    #[test]
    fn indexed_loads_through_built_descriptors_use_their_stride() {
        // No used-resource entry gives a stride: the descriptor's own (12,
        // and 1 as vertex fetch builds them) addresses the root view raw.
        for stride in [12, 1] {
            let l = indexed_root_load(stride << 16, 0x1101_4FAC).unwrap();
            assert_eq!(l.root_srvs, [3]);
        }
    }

    #[test]
    fn indexed_loads_through_swizzled_or_lane_indexed_descriptors_stop() {
        let why = |r: Result<Lowered, crate::ShaderError>| match r {
            Err(crate::ShaderError::Unsupported { what, .. }) => what,
            r => panic!("{:?}", r.map(|l| l.root_srvs)),
        };
        assert_eq!(
            why(indexed_root_load(0x8010_0000, 0x1101_4FAC)),
            "a swizzled buffer"
        );
        assert_eq!(
            why(indexed_root_load(0x0010_0000, 0x1181_4FAC)),
            "a buffer indexed by lane"
        );
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers_rsrc2(&words, 0).unwrap();
        }
    }

    /// An f16 (sign, 5-bit exponent, 10-bit fraction) as an f32; normal numbers only.
    fn f16_value(h: u32) -> f32 {
        let sign = if h >> 15 != 0 { -1.0 } else { 1.0 };
        let exponent = (h >> 10 & 0x1F) as i32 - 15;
        sign * (1.0 + (h & 0x3FF) as f32 / 1024.0) * 2f32.powi(exponent)
    }

    #[test]
    fn inline_float_constants_mean_the_same_in_f16() {
        // v_fma_mix and v_pack_b32_f16 read these as f16: 1.0 must be 0x3C00,
        // not the low half (0) of the f32 bits 0x3F800000.
        assert_eq!(inline_f16(242), 0x3C00);
        for code in 240..=248u8 {
            let f32_value = f32::from_bits(crate::resource::inline(code).unwrap());
            let f16_value = f16_value(inline_f16(code));
            assert!(
                (f32_value - f16_value).abs() <= f32_value.abs() * 1e-3,
                "inline constant {code}: f32 {f32_value}, f16 {f16_value}"
            );
        }
    }

    #[test]
    fn scalar_conditional_moves_lower() {
        // Each sets SCC, moves under it into a destination that already has a
        // value, and branches on the result (only code with an effect needs
        // its operands).
        let tail = [
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let b32 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xBE82_0501, // s_cmov_b32 s2, s1
        ];
        let k32 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xB102_0005, // s_cmovk_i32 s2, 5
        ];
        let b64 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE83_0387, // s_mov_b32 s3, 7
            0xBE80_0381, // s_mov_b32 s0, 1
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8000, // s_cmp_lg_u32 s0, 0
            0xBE82_0600, // s_cmov_b64 s[2:3], s[0:1]
        ];
        for head in [&b32[..], &k32[..], &b64[..]] {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words).unwrap();
        }
    }

    #[test]
    fn signed_names_of_wrapping_integer_operations_match_the_unsigned_ones() {
        // The low 32 bits of a sum, difference or product do not depend on
        // the sign, so the _i32 forms lower exactly as their _u32 twins.
        for (signed, unsigned) in [
            (0x37F, 0x125), // v_add_nc_i32, v_add_nc_u32
            (0x376, 0x126), // v_sub_nc_i32, v_sub_nc_u32
            (0x16B, 0x169), // v_mul_lo_i32, v_mul_lo_u32
        ] {
            let a = lower_ps(vop3(signed, 0, 2, 259, 129)).unwrap();
            let b = lower_ps(vop3(unsigned, 0, 2, 259, 129)).unwrap();
            assert_eq!(a, b, "opcode {signed:#x}");
        }
    }

    #[test]
    fn sixteen_bit_signed_add_and_sub_lower_with_half_selects() {
        // v_add_nc_i16 and v_sub_nc_i16, also reading the high halves.
        for op in [0x30D, 0x30E] {
            for opsel in [0, 1 << 11, 1 << 12 | 1 << 14] {
                lower_ps(vop3(op, opsel, 2, 259, 259)).unwrap();
            }
        }
    }

    #[test]
    fn clamped_signed_operations_are_refused_not_miscompiled() {
        // Clamping saturates; wrapping them would give other values.
        let clamp = 1 << 15;
        for op in [0x37F, 0x376, 0x16B, 0x30D, 0x30E] {
            let err = lower_ps(vop3(op, clamp, 2, 259, 129)).unwrap_err();
            assert!(
                matches!(err, ShaderError::Unsupported { what, .. } if what.contains("modifier")),
                "{op:#x}: {err:?}"
            );
        }
    }

    #[test]
    fn sub_dword_lds_names() {
        let s = |write, bits, signed, d16| {
            Some(SubDword {
                write,
                bits,
                signed,
                d16,
            })
        };
        assert_eq!(SubDword::of("ds_read_i8"), s(false, 8, true, None));
        assert_eq!(SubDword::of("ds_read_u16"), s(false, 16, false, None));
        assert_eq!(
            SubDword::of("ds_read_u8_d16"),
            s(false, 8, false, Some(false))
        );
        assert_eq!(
            SubDword::of("ds_read_i16_d16_hi"),
            s(false, 16, true, Some(true))
        );
        assert_eq!(SubDword::of("ds_write_b8"), s(true, 8, false, None));
        assert_eq!(
            SubDword::of("ds_write_b16_d16_hi"),
            s(true, 16, false, Some(true))
        );
        for name in [
            "ds_read_b32",
            "ds_read2_b32",
            "ds_write_b64",
            "ds_write_b8_d16",
            "ds_read_u8_d32",
            "ds_add_u32",
        ] {
            assert_eq!(SubDword::of(name), None, "{name}");
        }
    }

    #[test]
    fn scalar_pack_extend_and_absdiff_lower() {
        // Each computes into s2 from two constants and branches on it (only
        // code with an effect needs its operands).
        let head = [
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0383, // s_mov_b32 s1, 3
        ];
        let tail = [
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let ops = [
            0x9602_0100, // s_absdiff_i32 s2, s0, s1
            0xBE82_1901, // s_sext_i32_i8 s2, s1
            0xBE82_1A01, // s_sext_i32_i16 s2, s1
            0x9902_0100, // s_pack_ll_b32_b16 s2, s0, s1
            0x9982_0100, // s_pack_lh_b32_b16 s2, s0, s1
            0x9A02_0100, // s_pack_hh_b32_b16 s2, s0, s1
        ];
        for op in ops {
            let words: Vec<u32> = head.iter().chain(&[op]).chain(&tail).copied().collect();
            lowers(&words).unwrap();
        }
    }

    #[test]
    fn cube_descriptor_dwords_read_as_data() {
        // A cube image's descriptor (a table range at s0) is loaded, sized
        // with image_get_resinfo, and its depth dword (4: the six faces, so
        // 5) is read as a number and branched on.
        let table = || vec![root(RootType::Table, 0, 0)];
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], null
            0x7E04_0480, // v_mov_b32 v2, 0
            0xF038_0318,
            0x0001_0402, // image_get_resinfo v[4:5], v2, s[4:11] dim:SQ_RSRC_IMG_CUBE
            0xBF06_8508, // s_cmp_eq_u32 s8, 5
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        lower_with(&words, table(), vec![]).unwrap();
        // The same with a 2D image (dim 1) also lowers.
        let mut flat = words;
        flat[4] = 0xF038_0308;
        lower_with(&flat, table(), vec![]).unwrap();
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_BRANCHES_WAVE32, WATERFALL_CARRIED_WAVE32, WATERFALL_FETCHED_WAVE32,
            WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32, WATERFALL_REPLACED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
            (WATERFALL_CARRIED_WAVE32, false),
            (WATERFALL_BRANCHES_WAVE32, false),
            (WATERFALL_REPLACED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn waterfall_loops_over_descriptors_from_buffer_loads_sample_with_each_lanes_own() {
        let words = crate::resource::tests::waterfall_buffer_wave32();
        let (layout, used) = texture_tables();
        let l = lower_wave(&words, layout, used, false).unwrap();
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }

    #[test]
    fn per_lane_descriptors_loaded_under_a_partial_exec_fail() {
        // The if/else without its s_cbranch_execz: both branches run, each
        // loading for its lanes only, so v[4:11] after them is the second
        // load only in the else lanes.
        let mut words = crate::resource::tests::WATERFALL_BRANCHES_WAVE32.to_vec();
        for k in [12, 19] {
            assert_eq!(words[k] >> 16, 0xBF88);
            words[k] = 0xBF80_0000; // s_nop 0
        }
        let (layout, used) = texture_tables();
        let Err(e) = lower_wave(&words, layout, used, false) else {
            panic!("lowered");
        };
        assert!(
            matches!(
                e,
                crate::ShaderError::Unsupported {
                    what: "a per-lane descriptor loaded under a partial EXEC",
                    ..
                }
            ),
            "{e:?}"
        );
    }

    /// A table at s0 holding structured buffers of dwords.
    fn dword_buffers() -> Vec<UsedResource> {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    }

    #[test]
    fn descriptors_loaded_through_pointers_kept_in_phis() {
        // A buffer descriptor loaded from the table at s0 through a pointer
        // a root constant (s1) picks by a branch, then loaded through by
        // index and stored to LDS so the load is needed.
        let picked = [
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF85_0002, // s_cbranch_scc1 -> 0x10
            0x8004_9000, // s_add_u32 s4, s0, 16
            0xBF82_0001, // s_branch -> 0x14
            0x8004_A000, // 0x10: s_add_u32 s4, s0, 32
            0xBE85_0380, // 0x14: s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        // The same through a pointer advanced over four descriptors.
        let advanced = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0380, // s_mov_b32 s6, 0
            0xF408_0202,
            0xFA00_0000, // 0xc: s_load_dwordx4 s[8:11], s[4:5], null
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0x8004_9004, // s_add_u32 s4, s4, 16
            0x8006_8106, // s_add_u32 s6, s6, 1
            0xBF07_8406, // s_cmp_lg_u32 s6, 4
            0xBF85_FFF6, // s_cbranch_scc1 -> 0xc
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        lower_with(&picked, layout(), dword_buffers()).unwrap();
        lower_with(&advanced, layout(), dword_buffers()).unwrap();
    }

    #[test]
    fn data_read_through_a_pointer_into_a_root_descriptor_stops() {
        // A root descriptor's address (s[0:1]) advanced in a loop and read
        // through: its address is no value the translation has, so the
        // offset from it is unknown.
        let words = [
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0301, // s_mov_b32 s3, s1
            0xF400_0101,
            0xFA00_0000, // 0x8: s_load_dword s4, s[2:3], null
            0x8002_9002, // s_add_u32 s2, s2, 16
            0x8203_8003, // s_addc_u32 s3, s3, 0
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0xBF84_FFFA, // s_cbranch_scc0 -> 0x8
            S_ENDPGM,
        ];
        let e = lower_with(&words, vec![root(RootType::Descriptor, 0, 3)], vec![])
            .err()
            .unwrap();
        assert!(
            e.to_string()
                .contains("an advanced pointer into a root descriptor"),
            "{e}"
        );
    }

    #[test]
    fn scalar_zero_counts_and_bit_scans_lower() {
        // s_bcnt0_i32_b32/_b64, s_ff0_i32_b32/_b64, s_flbit_i32_b64 and
        // s_flbit_i32_i64 of s0 (or the pair s[0:1]), with a branch on the
        // result so the instruction has an effect.
        for op in [0x0D, 0x0E, 0x11, 0x12, 0x16, 0x18] {
            let words = [
                0xBE80_03C1,           // s_mov_b32 s0, -1
                0xBE81_0380,           // s_mov_b32 s1, 0
                0xBE82_0000 | op << 8, // <op> s2, s0 (s[0:1])
                0xBF07_8002,           // s_cmp_lg_u32 s2, 0
                0xBF85_0001,           // s_cbranch_scc1 -> s_endpgm
                0xBE83_0380,           // s_mov_b32 s3, 0
                S_ENDPGM,
            ];
            lowers(&words).unwrap_or_else(|e| panic!("SOP1 {op:#x}: {e}"));
        }
        // The zero counts also leave "result != 0" in SCC.
        for op in [0x0D, 0x0E] {
            let words = [
                0xBE80_03C1,           // s_mov_b32 s0, -1
                0xBE81_0380,           // s_mov_b32 s1, 0
                0xBE82_0000 | op << 8, // s_bcnt0_i32_b32/_b64 s2, s0 (s[0:1])
                0xBF85_0001,           // s_cbranch_scc1 -> s_endpgm
                0xBE83_0380,           // s_mov_b32 s3, 0
                S_ENDPGM,
            ];
            lowers(&words).unwrap_or_else(|e| panic!("SOP1 {op:#x} SCC: {e}"));
        }
    }

    #[test]
    fn private_memory_is_as_large_as_its_addresses_reach() {
        let bytes = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            super::scratch_bytes(&ir::build(&code, false, 256).unwrap())
        };
        let store = [
            0x7E02_0280, // v_mov_b32 v1, 0
            0xDC70_401C, // scratch_store_dword v1, v2, off offset:28
            0x007D_0201,
        ];
        let indexed = [
            0x3606_0087, // v_and_b32 v3, 7, v0
            0x3406_0682, // v_lshlrev_b32 v3, 2, v3
            0xDC30_4000, // scratch_load_dword v4, v3, off
            0x047D_0003,
        ];
        assert_eq!(bytes(&[&store[..], &indexed, &[S_ENDPGM]].concat()), 32);
        // An address the code does not bound (the thread id).
        let free = [0xDC30_4000, 0x047D_0000]; // scratch_load_dword v4, v0, off
        assert_eq!(
            bytes(&[&store[..], &free, &[S_ENDPGM]].concat()),
            super::SCRATCH_CAP
        );
        let far = [
            0x7E02_02FF, // v_mov_b32 v1, 0x100000
            0x0010_0000,
            0xDC70_4000, // scratch_store_dword v1, v2, off
            0x007D_0201,
            S_ENDPGM,
        ];
        assert_eq!(bytes(&far), super::SCRATCH_MAX);
    }

    #[test]
    fn generic_flat_memory_is_refused() {
        // flat_store_dword v[0:1], v2: the address may be private, shared
        // or global memory.
        let e = lowers(&[0xDC70_0000, 0x007D_0200, S_ENDPGM]).unwrap_err();
        assert!(
            matches!(
                e,
                crate::ShaderError::Unsupported {
                    what: "flat memory",
                    ..
                }
            ),
            "{e}"
        );
    }

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers_rsrc2(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers_rsrc2(&words, group_ids_xy).expect("dead timestamp");
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    /// Lowers a pixel program that loads a sampler and an image from two
    /// tables, samples v[0:3] from the address at v2 on with the two-dword
    /// instruction `sample`, and exports the result.
    fn lower_sample(sample: [u32; 2]) -> Result<Vec<u8>, crate::ShaderError> {
        let words = [
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0204,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
            0xF40C_0001,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
            sample[0],
            sample[1],
            0xF800_180F,
            0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
            0xBF81_0000, // s_endpgm
        ];
        let table = |user_data, slot| RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot,
        };
        let used = |kind, dwords, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![table(8, 0), table(9, 1)],
            used: vec![
                used(ResourceType::Sampler, 4, 8, 0),
                used(ResourceType::Srv, 8, 9, 1),
            ],
            io: StageIo::Ps {
                inputs: Vec::new(),
                input_addr: 0,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs)?;
        let res = crate::resource::track(&p, &native.layout, native.stage)?;
        let io = crate::io::raise(&p, &native)?;
        let st = crate::structure::structure(&p)?;
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: vec![target],
            prims: Vec::new(),
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
        .map(|l| l.dxil)
    }

    /// MIMG words: opcode, all four channels, the dimension; data v0,
    /// address v2, image s[0:7], sampler s[8:11].
    fn mimg(opcode: u32, dim: u32) -> [u32; 2] {
        [0xF000_0F00 | opcode << 18 | dim << 3, 0x0040_0002]
    }

    const SAMPLE_D: u32 = 0x22;
    const SAMPLE_D_CL: u32 = 0x23;
    const DIM_2D: u32 = 1;
    const DIM_CUBE: u32 = 3;

    #[test]
    fn gradient_samples_lower_on_planes_and_cubes() {
        for (opcode, dim) in [
            (SAMPLE_D, DIM_2D),
            (SAMPLE_D, DIM_CUBE),
            (SAMPLE_D_CL, DIM_2D),
            (SAMPLE_D_CL, DIM_CUBE),
        ] {
            if let Err(e) = lower_sample(mimg(opcode, dim)) {
                panic!("opcode {opcode:#x} on dimension {dim}: {e:?}");
            }
        }
    }

    #[test]
    fn host_wave32_mask_ops_use_both_ballot_halves() {
        use std::collections::HashMap;
        let words = [
            0xBE80_0F7E, // s_bcnt1_i32_b32 s0, exec_lo
            0xBE81_137E, // s_ff1_i32_b32 s1, exec_lo
            0xD765_0002,
            0x0001_007E, // v_mbcnt_lo_u32_b32 v2, exec_lo, 0
            0xD765_0003,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v3, -1, 0
            0xBF06_807E, // s_cmp_eq_u32 exec_lo, 0
            0x8504_0100, // s_cselect_b32 s4, s0, s1
            0x7E00_0204, // v_mov_b32 v0, s4
            0x7E02_0201, // v_mov_b32 v1, s1
            0xF800_180F,
            0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let p = ir::build(&native.code, false, 8).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let parts = Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        };
        let bad = HashMap::new();
        let mut l = super::Lower::new(&parts, &bad, false);
        assert!(l.lower_ir().is_ok());
        assert_eq!(l.mask_data, super::MaskData::Host);
        assert_eq!(l.s.wave_size, None);
        let &(_lo, hi) = l.ballots.values().next().unwrap();
        assert!(
            l.s.m.uses_of(hi) >= 4,
            "expected high ballot half to be used by s_bcnt1_i32_b32, s_ff1_i32_b32, v_mbcnt_lo_u32_b32, and s_cmp_eq_u32 (>= 4 uses), got {}",
            l.s.m.uses_of(hi)
        );
    }

    /// Lowers a one-thread compute program that copies four root constants
    /// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
    /// descriptor table, so the vector instructions in `body` have an effect.
    fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xFA00_0000,
        ];
        // v_mov_b32 v1..v4, s0..s3
        words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
        words.extend_from_slice(body);
        words.extend_from_slice(&[
            0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
            0x8002_0000,
            S_ENDPGM,
        ]);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let param = |root, sgprs, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                param(RootType::Constants, 4, 0, 0),
                param(RootType::Table, 1, 4, 1),
            ],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1, // five user SGPRs, s0..s4
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
        .map(|_| ())
    }

    #[test]
    fn the_24_bit_multiplies_lower_both_halves() {
        // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
        // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
        for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
            let word = op << 25 | 2 << 9 | 0x101;
            lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
        }
    }

    #[test]
    fn sums_of_absolute_differences_lower() {
        // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
        // v_sad_u32 and v_msad_u8.
        for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
            let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
            lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
        }
    }

    /// MIMG `op vdata, v[vaddr..], s[srsrc..srsrc+7]` on a 2D image.
    const fn mimg_2d(op: u8, dmask: u8, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0x3C << 26 | (op as u32 & 0x7F) << 18 | (dmask as u32) << 8 | 1 << 3,
            ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }

    #[test]
    fn an_image_read_as_float_and_by_an_atomic_is_annotated_twice() {
        use crate::native::{ResourceType, RootParam, RootType, UsedResource};

        // One 2D image UAV from the descriptor table at s4, loaded (four
        // float channels) and then atomically minimised in the same block.
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            // s_load_dwordx8 s[8:15], s[4:5], 0
            0x3D << 26 | 0x03 << 18 | 8 << 6 | 4 >> 1,
            125 << 25,
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
        ];
        words.extend(mimg_2d(0x00, 0xF, 2, 0, 8)); // image_load v[2:5], v[0:1], s[8:15]
        words.extend(mimg_2d(0x15, 0x1, 2, 0, 8)); // image_atomic_umin v2, v[0:1], s[8:15]
        words.push(S_ENDPGM);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 4,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let lowered = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap();

        // DXIL resource properties: word 0 is the kind (Texture2D = 2) with
        // the UAV bit (0x1000); word 1 is the component type in the low byte
        // (U32 = 5, F32 = 9, DxilConstants.h `ComponentType`) and the
        // component count above it. D3D12 allows atomics (Interlocked*) on
        // integer UAVs only, so the atomic needs its own one-U32 annotation
        // next to the load's four-float one.
        const TEXTURE_2D_UAV: u32 = 0x1002;
        assert!(
            lowered.annotations.contains(&(TEXTURE_2D_UAV, 0x409)),
            "the load's annotation: {:x?}",
            lowered.annotations
        );
        assert!(
            lowered.annotations.contains(&(TEXTURE_2D_UAV, 0x105)),
            "the atomic's annotation: {:x?}",
            lowered.annotations
        );
    }

    #[test]
    fn packed_half_inputs_pair_unlisted_rows() {
        use super::super::container::SigElement;
        use super::half_inputs;
        use crate::native::{PsInput, StageIo, Usage};

        let sig = |row: u32, half: bool| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half,
        };
        let ps_in = |row: u8, attr: u8, pack2: bool, row2: u8| PsInput {
            usage: Usage::Generic,
            row,
            mask: 3,
            attr,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        // Row 0 has row2 == row (0) and pairs with unlisted row 1;
        // row 2 explicitly names row2 == 4 (not stolen by row 0);
        // row 5 has row2 == row (5) and no remaining unlisted row after it.
        let inputs = vec![
            sig(0, true),
            sig(1, true),
            sig(2, true),
            sig(3, false),
            sig(4, true),
            sig(5, true),
        ];
        let io = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(2, 1, true, 4),
                ps_in(5, 3, true, 5),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map = half_inputs(&io, &inputs);
        assert_eq!(map.get(&0), Some(&(true, Some(1))));
        assert_eq!(map.get(&2), Some(&(true, Some(4))));
        assert_eq!(map.get(&5), Some(&(true, None)));

        // Duplicate entries for the same row keep the first entry and do not
        // consume an extra free row; an earlier unpaired row (0) must not steal
        // a free row (3) that comes after a later fp16 row (2).
        let inputs2 = vec![sig(0, true), sig(2, true), sig(3, true)];
        let io2 = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(0, 9, false, 0),
                ps_in(2, 1, true, 2),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map2 = half_inputs(&io2, &inputs2);
        assert_eq!(map2.get(&0), Some(&(true, None)));
        assert_eq!(map2.get(&2), Some(&(true, Some(3))));
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }

    /// From #305: 64-bit image atomics declare 64-bit integers and typed
    /// 64-bit atomics.
    #[test]
    fn image_atomics_64_bit_lower_with_typed_int64_flag() {
        use crate::native::{RootParam, RootType};

        let run = |mimg_w0: u32| {
            let words = [
                // s_mov_b32 s1, 0
                0xBE81_0380u32,
                // s_load_dwordx8 s[4:11], s[0:1], null
                0xF40C_0100,
                0xFA00_0000,
                // v_mov_b32 v0..v5, 0/1/2
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0281,
                0x7E06_0280,
                0x7E08_0282,
                0x7E0A_0280,
                // image_atomic_* (2D, unorm=1, glc=1) v[2..], v0, s[4:11]
                mimg_w0,
                0x0001_0200,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 8,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 8).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
        };

        // image_atomic_add (op 0x11), dim=1 (2D), unorm=1, dmask=3, glc=1
        let add = run(0xF044_1388).unwrap();
        // image_atomic_cmpswap (op 0x10), dim=1 (2D), unorm=1, dmask=0xf, glc=1
        let cmp = run(0xF040_1F88).unwrap();
        for lowered in [add, cmp] {
            let sfi0 = crate::dxbc::Container::parse(&lowered.dxil)
                .unwrap()
                .part(crate::dxbc::FourCC::SFI0)
                .unwrap()
                .data;
            let features = u64::from_le_bytes(sfi0.try_into().unwrap());
            assert_ne!(features & 0x8000, 0, "INT64 feature bit missing");
            assert_ne!(
                features & 0x40_0000,
                0,
                "ATOMIC_INT64_ON_TYPED_RESOURCE feature bit missing"
            );
        }
    }

    #[test]
    fn cube_and_msaa_descriptor_data_lower() {
        use crate::native::{RootParam, RootType};

        for dim in [3u32, 6, 7] {
            // s_mov_b32 s1, 0
            // s_load_dwordx8 s[4:11], s[0:1], null
            // s_lshr_b32 s12, s7, 16        (reads descriptor word 3 as data)
            // s_cmp_lg_u32 s12, 0
            // s_cbranch_scc1 -> s_endpgm
            // s_mov_b32 s13, 0
            // v_mov_b32 v0..v3, 0
            // image_get_resinfo (dim) v0, v0, s[4:11]  (op 0x0e, dmask=1)
            // s_endpgm
            let mimg_w0 = 0xF000_0000 | (0x0E << 18) | (1 << 8) | (dim << 3);
            let words = [
                0xBE81_0380u32,
                0xF40C_0100,
                0xFA00_0000,
                0x900C_9007,
                0xBF07_800C,
                0xBF85_0001,
                0xBE8D_0380,
                0x7E00_0280,
                mimg_w0,
                0x0001_0000,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            let out = lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            });
            // Cube (3) and MSAA (6, 7) descriptor_data() all lower without failing
            // on "an image dimension (cube or MSAA)".
            out.unwrap();
        }
    }

    #[test]
    fn msaa_image_load_and_get_resinfo_lower() {
        use crate::native::{RootParam, RootType};

        for (dim, op) in [(6u32, 0x00u32), (6, 0x0E), (7, 0x00), (7, 0x0E)] {
            // s_mov_b32 s1, 0
            // s_load_dwordx8 s[4:11], s[0:1], null
            // v_mov_b32 v0..v3, 0
            // image_load (0x00) / image_get_resinfo (0x0e) v[0..3], v[0..3], s[4:11]
            // v_cmp_ne_u32 vcc, v0, 0
            // s_cbranch_vccz -> s_endpgm
            // s_mov_b32 s12, 0
            // s_endpgm
            let mimg_w0 = 0xF000_0000 | (op << 18) | (0xF << 8) | (dim << 3);
            let words = [
                0xBE81_0380u32,
                0xF40C_0100,
                0xFA00_0000,
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                mimg_w0,
                0x0001_0000,
                0x7D9A_0080, // v_cmpx_ne_u32_e32 v0, 0 -> sets EXEC, effectful?
                0xBF81_0000,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }

    #[test]
    fn partial_channel_image_store_lowers() {
        use crate::native::{RootParam, RootType};

        // image_store (op 0x08), dim=1 (2D), unorm=1
        // Test 32-bit partial masks (0x1, 0x3, 0x5) and d16 partial mask (0x7).
        for (dmask, d16) in [(0x1u32, false), (0x3, false), (0x5, false), (0x7, true)] {
            let mimg_w0 = 0xF000_0000
                | (0x08 << 18)
                | (u32::from(d16) << 15)
                | (dmask << 8)
                | (1 << 7)
                | (1 << 3);
            let words = [
                // s_mov_b32 s1, 0
                0xBE81_0380u32,
                // s_load_dwordx8 s[4:11], s[0:1], null
                0xF40C_0100,
                0xFA00_0000,
                // v_mov_b32 v0..v3, 0 (coords v0..v1, data v2..v3)
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                // image_store v[2..], v[0..1], s[4:11]
                mimg_w0,
                0x0001_0200,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }

    #[test]
    fn partial_channel_typed_buffer_store_lowers() {
        use crate::native::{RootParam, RootType};

        // MUBUF buffer_store_format_{x,xy,xyz,d16_xyz} (ops 0x04, 0x05, 0x06, 0x86)
        // with idxen=1, vaddr=v0, vdata=v1, srsrc=s[4:7].
        for op in [0x04u32, 0x05, 0x06, 0x86] {
            let mubuf_w0 = 0xE000_2000 | (op << 18);
            let words = [
                // s_mov_b32 s1, 0
                0xBE81_0380u32,
                // s_load_dwordx4 s[4:7], s[0:1], null
                0xF408_0100,
                0xFA00_0000,
                // v_mov_b32 v0..v3, 0
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                // buffer_store_format_* v[1..], v0, s[4:7], 0 idxen
                mubuf_w0,
                0x8001_0100,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }
}

#[cfg(test)]
mod select_tests {
    use super::super::emit;
    use crate::native::{ComputeIo, RootParam};
    use crate::{Container, FourCC, HwStage, NativeProgram, ResourceType, RootType, UsedResource};

    #[test]
    fn selects_from_independently_written_mask_words() {
        let words = [
            0xbe81_0380u32, // s1 = 0 (table address high)
            0xf408_0200,
            0xfa00_0000, // load UAV into s[8:11]
            0xbe84_03ff,
            0x8000_0000, // s4 = lane 31
            0xbe85_0381, // s5 = lane 32 in wave64
            0x7e02_028b,
            0x7e04_029d, // v1 = 11, v2 = 29
            0xd501_0003,
            0x0012_0501, // v_cndmask_b32 v3, v1, v2, s[4:5]
            0x3402_0082, // v1 = 4 * v0
            0xe070_1000,
            0x8002_0301, // store v3 to UAV at v1
            0xbf81_0000,
        ];
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        for (wave64, uses_select) in [(false, true), (true, true), (true, false)] {
            let mut words = words;
            if !uses_select {
                words[12] = 0x8002_0101; // store v1 instead of the select result
            }
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                }],
                io: crate::native::StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [if wave64 { 64 } else { 32 }, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = crate::ir::build(&native.code, wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
            let shader = emit(&c, &native, &p, &st, &io, &res, crate::emit::Target::ANY).unwrap();
            let emitted = Container::parse(&shader.dxil).unwrap();
            let features = emitted.part(FourCC::SFI0).unwrap();
            let wave_ops = u64::from_le_bytes(features.data.try_into().unwrap()) & 0x4000 != 0;
            assert_eq!(wave_ops, uses_select);
        }
    }
}
