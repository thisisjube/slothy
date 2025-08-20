#
# Copyright (c) 2024 Justus Bergermann
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Justus Bergermann <mail@justus-bergermann.de>
#

"""This module contains abstract RISC-V instruction types to represent
instructions which share the same pattern"""
from slothy.targets.riscv.riscv import RegisterType
from slothy.targets.riscv.riscv_instruction_core import RISCVInstruction


# LMUL Helper Methods
def _get_lmul_value(obj=None):
    """Get LMUL value from instruction object or any loaded RISC-V target module"""
    import sys

    # Try to get from instruction object first
    if obj is not None:
        lmul = getattr(obj, "lmul", None)
        if lmul is not None:
            return _parse_lmul_string(lmul)

    # Try to get from any loaded RISC-V target module
    for module_name, module in sys.modules.items():
        if (
            module_name.startswith("slothy.targets.riscv.")
            and hasattr(module, "lmul")
            and module.lmul is not None
        ):
            return _parse_lmul_string(module.lmul)

    return 1  # Default


def _parse_lmul_string(lmul):
    """Parse LMUL string (e.g., 'm2', 'm4', 'm8', 'mf2', 'mf4', 'mf8') to integer"""
    if isinstance(lmul, str):
        if lmul.startswith("m") and not lmul.startswith("mf"):
            lmul = int(lmul[1:])  # e.g., "m2" -> 2
        elif lmul.startswith("mf"):
            lmul = 1  # Fractional LMUL, treat as 1 for now
        else:
            lmul = 1

    # Ensure LMUL is valid
    if lmul not in [1, 2, 4, 8]:
        lmul = 1

    return lmul


def _expand_vector_registers_generic(
    obj: any,
    expansion_factor: int,
    expansion_type: str = "lmul",
    num_vector_inputs: any = None,
    num_vector_outputs: any = None,
) -> any:
    """
    Expand vector registers based on expansion factor for vector instructions.

    Supports two expansion types:

    * **LMUL (Length Multiplier)**: Groups consecutive vector registers together

      - With ``LMUL=2``: ``v8`` becomes [``v8, v9``], ``v4`` becomes [``v4, v5``]
      - With ``LMUL=4``: ``v8`` becomes [``v8, v9, v10, v11``]

    * **NF (Number of Fields)**: Expands for load/store whole register operations

      - With ``NF=2``: ``v8`` becomes [``v8, v9``] for consecutive register groups

    This function:

    #. Automatically detects which operands are vector registers
    #. Expands vector operands into register groups
    #. Preserves scalar/immediate operands unchanged
    #. Sets up constraint combinations for SLOTHY's register allocator

    :param obj: Instruction object to modify
    :type obj: any
    :param expansion_factor: Expansion value (LMUL or NF value)
    :type expansion_factor: int
    :param expansion_type: Type of expansion (``"lmul"`` or ``"nf"``)
    :type expansion_type: str
    :param num_vector_inputs: Number of vector inputs (auto-detected if ``None``)
    :type num_vector_inputs: any
    :param num_vector_outputs: Number of vector outputs (auto-detected if ``None``)
    :type num_vector_outputs: any
    :return: modified obj
    :rtype: any
    """

    if expansion_factor <= 1:
        return obj

    available_regs = RegisterType.list_registers(RegisterType.VECT)

    def is_vector_register(reg):
        """Check if a register is a vector register."""
        return reg in available_regs

    def expand_vector_register(reg):
        """Expand a vector register into a group of consecutive registers."""
        if not is_vector_register(reg):
            return [reg]  # Not a vector register, keep as-is

        start_idx = available_regs.index(reg)
        if start_idx + expansion_factor > len(available_regs):
            return [reg]  # Not enough consecutive registers, keep original

        return [available_regs[start_idx + i] for i in range(expansion_factor)]

    def generate_combinations():
        """Generate all possible register group combinations."""
        combinations = []
        if expansion_type == "lmul":
            # LMUL requires aligned groups (start at multiples of expansion_factor)
            for i in range(0, len(available_regs), expansion_factor):
                if i + expansion_factor <= len(available_regs):
                    combinations.append(
                        [available_regs[i + j] for j in range(expansion_factor)]
                    )
        else:  # nf
            # NF allows any consecutive group
            for i in range(0, len(available_regs) - expansion_factor + 1):
                combinations.append(
                    [available_regs[i + j] for j in range(expansion_factor)]
                )
        return combinations

    # Save original state for reference
    orig_args_in = obj.args_in.copy()
    orig_args_out = obj.args_out.copy()
    orig_arg_types_in = obj.arg_types_in.copy()
    orig_arg_types_out = obj.arg_types_out.copy()

    # === ANALYZE OPERAND TYPES ===
    # Auto-detect vector operands if not specified
    if num_vector_outputs is None:
        num_vector_outputs = sum(1 for reg in orig_args_out if is_vector_register(reg))

    if num_vector_inputs is None:
        num_vector_inputs = sum(1 for reg in orig_args_in if is_vector_register(reg))

    # Identify which operand positions are vectors
    vector_output_indices = [
        i for i, reg in enumerate(orig_args_out) if is_vector_register(reg)
    ]
    vector_input_indices = [
        i for i, reg in enumerate(orig_args_in) if is_vector_register(reg)
    ]

    # === EXPAND OUTPUT REGISTERS ===
    expanded_outputs = []
    new_arg_types_out = []

    for i, reg in enumerate(orig_args_out):
        if i in vector_output_indices:
            expanded_regs = expand_vector_register(reg)
            expanded_outputs.extend(expanded_regs)
            new_arg_types_out.extend([RegisterType.VECT] * len(expanded_regs))
        else:
            expanded_outputs.append(reg)
            new_arg_types_out.append(orig_arg_types_out[i])

    # === EXPAND INPUT REGISTERS ===
    expanded_inputs = []
    new_arg_types_in = []

    for i, reg in enumerate(orig_args_in):
        if i in vector_input_indices:
            expanded_regs = expand_vector_register(reg)
            expanded_inputs.extend(expanded_regs)
            new_arg_types_in.extend([RegisterType.VECT] * len(expanded_regs))
        else:
            expanded_inputs.append(reg)
            new_arg_types_in.append(orig_arg_types_in[i])

    # === UPDATE INSTRUCTION OBJECT ===
    obj.args_out = expanded_outputs
    obj.args_in = expanded_inputs
    obj.num_out = len(expanded_outputs)
    obj.num_in = len(expanded_inputs)
    obj.arg_types_out = new_arg_types_out
    obj.arg_types_in = new_arg_types_in

    # === SET UP REGISTER ALLOCATION CONSTRAINTS ===
    valid_combinations = generate_combinations()

    # Calculate constraint indices for expanded vector operands
    vector_output_constraint_indices = []
    vector_input_constraint_indices = []

    # Map expanded vector outputs to constraint indices
    expanded_idx = 0
    for i, reg in enumerate(orig_args_out):
        if i in vector_output_indices:
            group_size = len(expand_vector_register(reg))
            vector_output_constraint_indices.extend(
                range(expanded_idx, expanded_idx + group_size)
            )
            expanded_idx += group_size
        else:
            expanded_idx += 1

    # Map expanded vector inputs to constraint indices
    expanded_idx = 0
    for i, reg in enumerate(orig_args_in):
        if i in vector_input_indices:
            group_size = len(expand_vector_register(reg))
            vector_input_constraint_indices.extend(
                range(expanded_idx, expanded_idx + group_size)
            )
            expanded_idx += group_size
        else:
            expanded_idx += 1

    # Output constraints: only apply to vector outputs
    if vector_output_constraint_indices:
        obj.args_out_combinations = [
            (vector_output_constraint_indices, valid_combinations)
        ]

    # Input constraints: only apply to vector inputs
    if vector_input_constraint_indices:
        if num_vector_inputs == 1:
            # Single vector input: use simple combinations
            obj.args_in_combinations = [
                (vector_input_constraint_indices, valid_combinations)
            ]
        else:
            # Multiple vector inputs: generate cartesian product of valid groups
            import itertools

            multi_combinations = [
                [reg for combo in combination for reg in combo]
                for combination in itertools.product(
                    valid_combinations, repeat=num_vector_inputs
                )
            ]
            obj.args_in_combinations = [
                (vector_input_constraint_indices, multi_combinations)
            ]

    # Set up empty restrictions (no specific register restrictions)
    obj.args_out_restrictions = [None] * obj.num_out if obj.num_out > 0 else []
    obj.args_in_restrictions = [None] * obj.num_in if obj.num_in > 0 else []

    return obj


def _expand_vector_registers_for_lmul(
    obj, lmul, num_vector_inputs=None, num_vector_outputs=None
):
    """Backward compatibility wrapper for LMUL expansion."""
    return _expand_vector_registers_generic(
        obj, lmul, "lmul", num_vector_inputs, num_vector_outputs
    )


def _expand_vector_registers_for_nf(obj, nf, load_or_store="load"):
    """Expand vector registers for load/store whole register instructions using NF."""
    # Use the generic function and let it auto-detect vector operands
    # The function will correctly identify which operands are vectors vs scalars
    return _expand_vector_registers_generic(obj, nf, "nf")


def _write_lmul_instruction(self, lmul, num_vector_inputs):
    """Custom write method for LMUL instructions that shows only base registers"""
    if lmul > 1 and len(self.args_out) > 1:
        out = self.pattern
        # Use only the first register from each group for display
        display_args_out = [self.args_out[0]] if self.args_out else []
        display_args_in = []

        # Extract first register from each vector input group
        input_idx = 0
        for _ in range(num_vector_inputs):
            if input_idx < len(self.args_in):
                display_args_in.append(self.args_in[input_idx])
                input_idx += lmul

        # Add any remaining non-vector inputs
        while input_idx < len(self.args_in):
            display_args_in.append(self.args_in[input_idx])
            input_idx += 1

        l = (
            list(zip(display_args_in, self.pattern_inputs))
            + list(zip(display_args_out, self.pattern_outputs))
            + list(zip(self.args_in_out, self.pattern_in_outs))
        )

        for arg, (s, ty) in l:
            out = RISCVInstruction._instantiate_pattern(s, ty, arg, out)

        # Handle other pattern replacements
        def replace_pattern(txt, attr_name, mnemonic_key, t=None):
            def t_default(x):
                return x

            if t is None:
                t = t_default
            a = getattr(self, attr_name)
            if a is None and attr_name == "is32bit":
                return txt.replace("<w>", "")
            if a is None:
                return txt
            if not isinstance(a, list):
                txt = txt.replace(f"<{mnemonic_key}>", t(a))
                return txt
            for i, v in enumerate(a):
                txt = txt.replace(f"<{mnemonic_key}{i}>", t(v))
            return txt

        out = replace_pattern(out, "immediate", "imm", lambda x: f"{x}")
        out = replace_pattern(out, "datatype", "dt", lambda x: x.upper())
        out = replace_pattern(out, "flag", "flag")
        out = replace_pattern(out, "index", "index", str)
        out = replace_pattern(out, "is32bit", "w", lambda x: x.lower())
        out = replace_pattern(out, "len", "len")
        out = replace_pattern(out, "vm", "vm")
        out = replace_pattern(out, "vtype", "vtype")
        out = replace_pattern(out, "sew", "sew")
        out = replace_pattern(out, "lmul", "lmul")
        out = replace_pattern(out, "tpol", "tpol")
        out = replace_pattern(out, "mpol", "mpol")
        out = replace_pattern(out, "nf", "nf")
        out = replace_pattern(out, "ew", "ew")

        out = out.replace("\\[", "[")
        out = out.replace("\\]", "]")
        return out
    else:
        # Default write behavior for LMUL=1
        return RISCVInstruction.write(self)


class RISCVStore(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        obj.pre_index = obj.immediate
        obj.addr = obj.args_in[1]
        return obj

    pattern = "mnemonic <Xb>, <imm>(<Xa>)"
    inputs = ["Xb", "Xa"]


# Scalar instructions


class RISCVIntegerRegister(RISCVInstruction):
    pattern = "mnemonic <Xd>, <Xa>"
    inputs = ["Xa"]
    outputs = ["Xd"]


class RISCVIntegerRegisterImmediate(RISCVInstruction):
    pattern = "mnemonic <Xd>, <Xa>, <imm>"
    inputs = ["Xa"]
    outputs = ["Xd"]


class RISCVIntegerRegisterRegister(RISCVInstruction):
    pattern = "mnemonic <Xd>, <Xa>, <Xb>"
    inputs = ["Xa", "Xb"]
    outputs = ["Xd"]


class RISCVLoad(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

    pattern = "mnemonic <Xd>, <imm>(<Xa>)"
    inputs = ["Xa"]
    outputs = ["Xd"]


class RISCVUType(RISCVInstruction):
    pattern = "mnemonic <Xd>, <imm>"
    outputs = ["Xd"]


class RISCVIntegerRegisterRegisterMul(RISCVInstruction):
    pattern = "mnemonic <Xd>, <Xa>, <Xb>"
    inputs = ["Xa", "Xb"]
    outputs = ["Xd"]


# Vector instructions ####

# Load Instructions ##


class RISCVVectorLoadUnitStride(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

    pattern = "mnemonic <Vd>, (<Xa>)<vm>"
    inputs = ["Xa"]
    outputs = ["Vd"]


class RISCVVectorLoadStrided(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]
    outputs = ["Vd"]


class RISCVVectorLoadIndexed(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

    pattern = "mnemonic <Vd>, (<Xa>), <Ve><vm>"
    inputs = ["Xa", "Ve"]
    outputs = ["Vd"]


class RISCVVectorLoadWholeRegister(RISCVInstruction):
    def write(self):
        out = self.pattern
        l = (
            list(zip(self.args_in, self.pattern_inputs))
            + list(zip(self.args_out, self.pattern_outputs))
            + list(zip(self.args_in_out, self.pattern_in_outs))
        )

        for arg, (s, ty) in l[:2]:
            out = RISCVInstruction._instantiate_pattern(s, ty, arg, out)

        def replace_pattern(txt, attr_name, mnemonic_key, t=None):
            def t_default(x):
                return x

            if t is None:
                t = t_default

            a = getattr(self, attr_name)
            if a is None and attr_name == "is32bit":
                return txt.replace("<w>", "")
            if a is None:
                return txt
            if not isinstance(a, list):
                txt = txt.replace(f"<{mnemonic_key}>", t(a))
                return txt
            for i, v in enumerate(a):
                txt = txt.replace(f"<{mnemonic_key}{i}>", t(v))
            return txt

        out = replace_pattern(out, "immediate", "imm", lambda x: f"{x}")
        out = replace_pattern(out, "datatype", "dt", lambda x: x.upper())
        out = replace_pattern(out, "flag", "flag")
        out = replace_pattern(out, "index", "index", str)
        out = replace_pattern(out, "is32bit", "w", lambda x: x.lower())
        out = replace_pattern(out, "len", "len")
        out = replace_pattern(out, "vm", "vm")
        out = replace_pattern(out, "vtype", "vtype")
        out = replace_pattern(out, "sew", "sew")
        out = replace_pattern(out, "lmul", "lmul")
        out = replace_pattern(out, "tpol", "tpol")
        out = replace_pattern(out, "mpol", "mpol")
        out = replace_pattern(out, "nf", "nf")
        out = replace_pattern(out, "ew", "ew")

        out = out.replace("\\[", "[")
        out = out.replace("\\]", "]")
        return out

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]

        # Use the original _expand_reg method for compatibility
        regs_types, expanded_regs = RISCVInstruction._expand_reg(
            obj.args_out[0], obj.nf, "load"
        )
        obj.args_out = expanded_regs
        obj.num_out = len(obj.args_out)
        obj.arg_types_out = regs_types

        # Set up register allocation constraints using the generalized approach
        available_regs = RegisterType.list_registers(RegisterType.VECT)
        combinations = []
        # NF allows any consecutive group
        for i in range(0, len(available_regs) - int(obj.nf) + 1):
            combinations.append([available_regs[i + j] for j in range(int(obj.nf))])

        obj.args_out_combinations = [(list(range(0, int(obj.num_out))), combinations)]
        obj.args_out_restrictions = [None for _ in range(obj.num_out)]

        # Update pattern outputs for the expanded registers
        vlist = [
            "V" + chr(i) for i in range(ord("d"), ord("z") + 1)
        ]  # list of all V registers names
        obj.outputs = vlist[: int(obj.nf)]
        obj.pattern_outputs = list(zip(obj.outputs, obj.arg_types_out))
        return obj

    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Xa"]
    outputs = ["Vd"]


# Store Instructions ##


class RISCVVectorStoreUnitStride(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

    pattern = "mnemonic <Va>, (<Xa>)<vm>"
    inputs = ["Xa", "Va"]
    outputs = []


class RISCVVectorStoreStrided(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

    pattern = "mnemonic <Vd>, (<Xa>), <Xb><vm>"
    inputs = ["Xa", "Xb"]


class RISCVVectorStoreIndexed(RISCVInstruction):
    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[0]
        return obj

    pattern = "mnemonic <Vd>, (<Xa>), <Ve><vm>"
    inputs = ["Xa", "Ve"]


class RISCVVectorStoreWholeRegister(RISCVInstruction):
    def write(self):
        out = self.pattern
        l = (
            list(zip(self.args_in, self.pattern_inputs))
            + list(zip(self.args_out, self.pattern_outputs))
            + list(zip(self.args_in_out, self.pattern_in_outs))
        )

        for arg, (s, ty) in [l[-1], l[0]]:
            out = RISCVInstruction._instantiate_pattern(s, ty, arg, out)

        def replace_pattern(txt, attr_name, mnemonic_key, t=None):
            def t_default(x):
                return x

            if t is None:
                t = t_default

            a = getattr(self, attr_name)
            if a is None and attr_name == "is32bit":
                return txt.replace("<w>", "")
            if a is None:
                return txt
            if not isinstance(a, list):
                txt = txt.replace(f"<{mnemonic_key}>", t(a))
                return txt
            for i, v in enumerate(a):
                txt = txt.replace(f"<{mnemonic_key}{i}>", t(v))
            return txt

        out = replace_pattern(out, "immediate", "imm", lambda x: f"{x}")
        out = replace_pattern(out, "datatype", "dt", lambda x: x.upper())
        out = replace_pattern(out, "flag", "flag")
        out = replace_pattern(out, "index", "index", str)
        out = replace_pattern(out, "is32bit", "w", lambda x: x.lower())
        out = replace_pattern(out, "len", "len")
        out = replace_pattern(out, "vm", "vm")
        out = replace_pattern(out, "vtype", "vtype")
        out = replace_pattern(out, "sew", "sew")
        out = replace_pattern(out, "lmul", "lmul")
        out = replace_pattern(out, "tpol", "tpol")
        out = replace_pattern(out, "mpol", "mpol")
        out = replace_pattern(out, "nf", "nf")
        out = replace_pattern(out, "ew", "ew")

        out = out.replace("\\[", "[")
        out = out.replace("\\]", "]")
        return out

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        # obj.pre_index = obj.immediate
        obj.addr = obj.args_in[1]

        # Use the original _expand_reg method for compatibility
        regs_types, expanded_regs = RISCVInstruction._expand_reg(
            obj.args_in[0], obj.nf, "store"
        )
        mem_reg = obj.args_in[1]
        obj.args_in = expanded_regs + [
            mem_reg
        ]  # add the register holding the memory address
        obj.num_in = len(obj.args_in)
        obj.arg_types_in = regs_types

        # Set up register allocation constraints using the generalized approach
        available_regs = RegisterType.list_registers(RegisterType.VECT)
        combinations = []
        # NF allows any consecutive group
        for i in range(0, len(available_regs) - int(obj.nf) + 1):
            combinations.append([available_regs[i + j] for j in range(int(obj.nf))])

        obj.args_in_combinations = [(list(range(0, int(obj.num_in - 1))), combinations)]
        obj.args_in_restrictions = [None for _ in range(obj.num_in)]

        vlist = [
            "V" + chr(i) for i in range(ord("d"), ord("z") + 1)
        ]  # list of all V registers names
        obj.inputs = vlist[: int(obj.nf)] + ["Xa"]
        obj.pattern_inputs = list(zip(obj.inputs, obj.arg_types_in))

        return obj

    pattern = "mnemonic <Vd>, (<Xa>)"
    inputs = ["Vd", "Xa"]


# Vector Integer Instructions ##


class RISCVVectorIntegerVectorVector(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Ve>, <Vf><vm>"
    inputs = ["Ve", "Vf"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=2)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        return _expand_vector_registers_for_lmul(obj, lmul)


# mask is fixed to v0
class RISCVVectorIntegerVectorVectorMasked(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Ve>, <Vf>, <Vg>"  # Vg == v0
    inputs = ["Ve", "Vf", "Vg"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=2)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Note: mask register (Vg) is not expanded, only vector operands
        return _expand_vector_registers_for_lmul(obj, lmul)


class RISCVVectorIntegerVectorScalar(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Ve>, <Xa><vm>"
    inputs = ["Ve", "Xa"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=1)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Only the vector input is expanded, scalar input (Xa) is kept as-is
        return _expand_vector_registers_for_lmul(obj, lmul)


# mask is fixed to v0
class RISCVVectorIntegerVectorScalarMasked(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Ve>, <Xa>, <Vg>"  # Vg == v0
    inputs = ["Ve", "Xa", "Vg"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=1)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Only the vector input is expanded, scalar (Xa) and mask (Vg) are kept as-is
        return _expand_vector_registers_for_lmul(obj, lmul)


class RISCVVectorIntegerVectorImmediate(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Ve>, <imm><vm>"
    inputs = ["Ve"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=1)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Only the vector input is expanded
        return _expand_vector_registers_for_lmul(obj, lmul)


# mask is fixed to v0
class RISCVVectorIntegerVectorImmediateMasked(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Ve>, <imm>, <Vg>"
    inputs = ["Ve", "Vg"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=1)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Only the vector input is expanded, mask (Vg) is kept as-is
        return _expand_vector_registers_for_lmul(obj, lmul)


# Vector Permutation Instructions


class RISCVScalarVector(RISCVInstruction):
    pattern = "mnemonic <Xd>, <Ve>"
    inputs = ["Ve"]
    outputs = ["Xd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=1)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Vector input is expanded, scalar output (Xd) is kept as-is
        return _expand_vector_registers_for_lmul(obj, lmul)


class RISCVVectorScalar(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Xa>"
    inputs = ["Xa"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=0)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Scalar input (Xa) is kept as-is, vector output is expanded
        return _expand_vector_registers_for_lmul(obj, lmul)


class RISCVVectorVector(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Va>"
    inputs = ["Va"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=1)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Both vector input and output are expanded
        return _expand_vector_registers_for_lmul(obj, lmul)


class RISCVectorVectorMasked(RISCVInstruction):
    pattern = "mnemonic <Vd>, <Va><vm>"
    inputs = ["Va"]
    outputs = ["Vd"]

    def write(self):
        lmul = _get_lmul_value(self)
        return _write_lmul_instruction(self, lmul, num_vector_inputs=1)

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        lmul = _get_lmul_value(obj)
        # Both vector input and output are expanded, mask is implicit in pattern
        return _expand_vector_registers_for_lmul(obj, lmul)


class RISCVBranch(RISCVInstruction):
    """RISC-V branch instructions with two register operands and a label"""

    pattern = "mnemonic <Xa>, <Xb>, <label>"
    inputs = ["Xa", "Xb"]
    outputs = []

    @classmethod
    def make(cls, src):
        obj = RISCVInstruction.build(cls, src)
        obj.increment = None
        obj.immediate = None
        # Initialize label attribute to avoid AttributeError
        if not hasattr(obj, "label"):
            obj.label = None
        return obj
