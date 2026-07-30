"""Allocation-free kernels exposed to the Python bindings through a C ABI."""

from std.algorithm import parallelize
from std.bit import pop_count
from std.math import sqrt
from std.sys.info import simd_width_of

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def hash_combine(seed: UInt32, value: UInt32) -> UInt32:
    return seed ^ (value + UInt32(0x9E3779B9) + (seed << 6) + (seed >> 2))


def pair_hash(first: Int32, second: UInt32) -> UInt32:
    return hash_combine(hash_combine(UInt32(0), UInt32(first)), second)


def same_mask(a: U8Ptr, b: U8Ptr, n: Int) -> Bool:
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        if a.load[width=W](i) != b.load[width=W](i):
            return False
        i += W
    while i < n:
        if a[i] != b[i]:
            return False
        i += 1
    return True


@export("mrd_morgan")
def mrd_morgan(
    atom_inv_addr: Int,
    bond_inv_addr: Int,
    offsets_addr: Int,
    neighbors_addr: Int,
    neighbor_bonds_addr: Int,
    chiral_tags_addr: Int,
    cip_codes_addr: Int,
    include_atoms_addr: Int,
    n_atoms: Int,
    n_bonds: Int,
    radius: Int,
    include_redundant: Int,
    codes_addr: Int,
    centers_addr: Int,
    layers_addr: Int,
    current_addr: Int,
    next_layer_addr: Int,
    dead_addr: Int,
    active_addr: Int,
    atom_hoods_addr: Int,
    round_hoods_addr: Int,
    env_hoods_addr: Int,
    chiral_seen_addr: Int,
    pair_bonds_addr: Int,
    pair_invariants_addr: Int,
    result_words_addr: Int,
    n_bits: Int,
) abi("C") -> Int:
    var atom_inv = U32Ptr(unsafe_from_address=atom_inv_addr)
    var bond_inv = I32Ptr(unsafe_from_address=bond_inv_addr)
    var offsets = I32Ptr(unsafe_from_address=offsets_addr)
    var neighbors = I32Ptr(unsafe_from_address=neighbors_addr)
    var neighbor_bonds = I32Ptr(unsafe_from_address=neighbor_bonds_addr)
    var chiral_tags = U8Ptr(unsafe_from_address=chiral_tags_addr)
    var cip_codes = U8Ptr(unsafe_from_address=cip_codes_addr)
    var include_atoms = U8Ptr(unsafe_from_address=include_atoms_addr)
    var codes = U32Ptr(unsafe_from_address=codes_addr)
    var centers = I32Ptr(unsafe_from_address=centers_addr)
    var layers = I32Ptr(unsafe_from_address=layers_addr)
    var current = U32Ptr(unsafe_from_address=current_addr)
    var next_layer = U32Ptr(unsafe_from_address=next_layer_addr)
    var dead = U8Ptr(unsafe_from_address=dead_addr)
    var active = U8Ptr(unsafe_from_address=active_addr)
    var atom_hoods = U8Ptr(unsafe_from_address=atom_hoods_addr)
    var round_hoods = U8Ptr(unsafe_from_address=round_hoods_addr)
    var env_hoods = U8Ptr(unsafe_from_address=env_hoods_addr)
    var chiral_seen = U8Ptr(unsafe_from_address=chiral_seen_addr)
    var pair_bonds = I32Ptr(unsafe_from_address=pair_bonds_addr)
    var pair_invariants = U32Ptr(unsafe_from_address=pair_invariants_addr)
    var result_words = U64Ptr(unsafe_from_address=result_words_addr)

    comptime W = simd_width_of[DType.float64]()
    var n_words = (n_bits + 63) // 64
    var word = 0
    var zero_words = SIMD[DType.uint64, W](0)
    while word + W <= n_words:
        result_words.store(word, zero_words)
        word += W
    while word < n_words:
        result_words[word] = UInt64(0)
        word += 1
    var count = 0
    for i in range(n_atoms):
        current[i] = atom_inv[i]
        dead[i] = UInt8(0)
        chiral_seen[i] = UInt8(0)
        for b in range(n_bonds):
            atom_hoods[i * n_bonds + b] = UInt8(0)
        if include_atoms[i] != 0:
            codes[count] = current[i]
            centers[count] = Int32(i)
            layers[count] = Int32(0)
            var bit = Int(current[i]) % n_bits
            result_words[bit >> 6] = (
                result_words[bit >> 6] | (UInt64(1) << UInt64(bit & 63))
            )
            count += 1

    for layer in range(radius):
        for i in range(n_atoms):
            active[i] = UInt8(0)
            next_layer[i] = UInt32(0)
            for b in range(n_bonds):
                round_hoods[i * n_bonds + b] = atom_hoods[i * n_bonds + b]

        for atom in range(n_atoms):
            if dead[atom] != 0:
                continue
            var begin = Int(offsets[atom])
            var end = Int(offsets[atom + 1])
            if begin == end:
                dead[atom] = UInt8(1)
                continue

            var degree = 0
            for edge_pos in range(begin, end):
                var other = Int(neighbors[edge_pos])
                var bond = Int(neighbor_bonds[edge_pos])
                round_hoods[atom * n_bonds + bond] = UInt8(1)
                for b in range(n_bonds):
                    if atom_hoods[other * n_bonds + b] != 0:
                        round_hoods[atom * n_bonds + b] = UInt8(1)
                pair_bonds[degree] = bond_inv[bond]
                pair_invariants[degree] = current[other]
                degree += 1

            for j in range(1, degree):
                var key_bond = pair_bonds[j]
                var key_inv = pair_invariants[j]
                var k = j
                while k > 0 and (
                    pair_bonds[k - 1] > key_bond or (
                        pair_bonds[k - 1] == key_bond
                        and pair_invariants[k - 1] > key_inv
                    )
                ):
                    pair_bonds[k] = pair_bonds[k - 1]
                    pair_invariants[k] = pair_invariants[k - 1]
                    k -= 1
                pair_bonds[k] = key_bond
                pair_invariants[k] = key_inv

            var invariant = hash_combine(UInt32(layer), current[atom])
            var looks_chiral = chiral_tags[atom] != 0
            for j in range(degree):
                invariant = hash_combine(
                    invariant, pair_hash(pair_bonds[j], pair_invariants[j])
                )
                if looks_chiral and chiral_seen[atom] == 0:
                    if pair_bonds[j] != 1:
                        looks_chiral = False
                    elif j > 0 and pair_invariants[j] == pair_invariants[j - 1]:
                        looks_chiral = False
            if looks_chiral:
                chiral_seen[atom] = UInt8(1)
                invariant = hash_combine(invariant, UInt32(cip_codes[atom]))

            next_layer[atom] = invariant
            active[atom] = UInt8(1)

        for atom in range(n_atoms):
            if active[atom] == 0:
                continue
            var duplicate = False
            if include_redundant == 0:
                for previous in range(count):
                    if layers[previous] != 0 and same_mask(
                        round_hoods + atom * n_bonds,
                        env_hoods + previous * n_bonds,
                        n_bonds,
                    ):
                        duplicate = True
                        break
                if not duplicate:
                    for other in range(n_atoms):
                        if other == atom or active[other] == 0 or include_atoms[other] == 0:
                            continue
                        if same_mask(
                            round_hoods + atom * n_bonds,
                            round_hoods + other * n_bonds,
                            n_bonds,
                        ):
                            if next_layer[other] < next_layer[atom] or (
                                next_layer[other] == next_layer[atom] and other < atom
                            ):
                                duplicate = True
                                break
            if duplicate:
                dead[atom] = UInt8(1)
            elif include_atoms[atom] != 0:
                codes[count] = next_layer[atom]
                centers[count] = Int32(atom)
                layers[count] = Int32(layer + 1)
                var bit = Int(next_layer[atom]) % n_bits
                result_words[bit >> 6] = (
                    result_words[bit >> 6] | (UInt64(1) << UInt64(bit & 63))
                )
                for b in range(n_bonds):
                    env_hoods[count * n_bonds + b] = round_hoods[atom * n_bonds + b]
                count += 1

        for atom in range(n_atoms):
            current[atom] = next_layer[atom]
            for b in range(n_bonds):
                atom_hoods[atom * n_bonds + b] = round_hoods[atom * n_bonds + b]

    return count


def similarity_value(common: Int, left: Int, right: Int, n_bits: Int, metric: Int) -> Float64:
    if left == 0 or right == 0:
        return 0.0
    var c = Float64(common)
    var a = Float64(left)
    var b = Float64(right)
    if metric == 0:
        return c / (a + b - c)
    if metric == 1:
        return 2.0 * c / (a + b)
    if metric == 2:
        return c / sqrt(a * b)
    if metric == 3:
        return c / (2.0 * a + 2.0 * b - 3.0 * c)
    if metric == 4:
        return c / Float64(n_bits)
    if metric == 5:
        return c * (a + b) / (2.0 * a * b)
    return c / a + c / b - 1.0


@export("mrd_similarity")
def mrd_similarity(
    left_addr: Int,
    right_addr: Int,
    n_words: Int,
    n_bits: Int,
    metric: Int,
) abi("C") -> Float64:
    var left_words = U64Ptr(unsafe_from_address=left_addr)
    var right_words = U64Ptr(unsafe_from_address=right_addr)
    var left_count = 0
    var right_count = 0
    var common = 0
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n_words:
        var left_values = left_words.load[width=W](i)
        var right_values = right_words.load[width=W](i)
        left_count += Int(pop_count(left_values).reduce_add())
        right_count += Int(pop_count(right_values).reduce_add())
        common += Int(pop_count(left_values & right_values).reduce_add())
        i += W
    while i < n_words:
        left_count += Int(pop_count(left_words[i]))
        right_count += Int(pop_count(right_words[i]))
        common += Int(pop_count(left_words[i] & right_words[i]))
        i += 1
    return similarity_value(common, left_count, right_count, n_bits, metric)


@export("mrd_bulk_similarity")
def mrd_bulk_similarity(
    query_addr: Int,
    targets_addr: Int,
    scores_addr: Int,
    n_targets: Int,
    n_words: Int,
    n_bits: Int,
    metric: Int,
) abi("C"):
    var query = U64Ptr(unsafe_from_address=query_addr)
    var targets = U64Ptr(unsafe_from_address=targets_addr)
    var scores = F64Ptr(unsafe_from_address=scores_addr)
    var query_count = 0
    comptime W = simd_width_of[DType.float64]()
    var w = 0
    while w + W <= n_words:
        query_count += Int(pop_count(query.load[width=W](w)).reduce_add())
        w += W
    while w < n_words:
        query_count += Int(pop_count(query[w]))
        w += 1

    @parameter
    def compute_row(row: Int):
        var target_count = 0
        var common = 0
        var column = 0
        var row_offset = row * n_words
        while column + W <= n_words:
            var values = targets.load[width=W](row_offset + column)
            var query_values = query.load[width=W](column)
            target_count += Int(pop_count(values).reduce_add())
            common += Int(pop_count(query_values & values).reduce_add())
            column += W
        while column < n_words:
            var value = targets[row_offset + column]
            target_count += Int(pop_count(value))
            common += Int(pop_count(query[column] & value))
            column += 1
        scores[row] = similarity_value(common, query_count, target_count, n_bits, metric)

    @parameter
    def compute_chunk(chunk: Int):
        var begin = (chunk * n_targets) // 8
        var end = ((chunk + 1) * n_targets) // 8
        for row in range(begin, end):
            compute_row(row)

    if n_targets * n_words >= 1048576:
        parallelize[compute_chunk](8, 8)
    else:
        for row in range(n_targets):
            compute_row(row)


def contains_mapping(
    matches: I32Ptr, mapping: I32Ptr, count: Int, n_query: Int
) -> Bool:
    var mapping_sum = 0
    var mapping_square_sum = 0
    for q in range(n_query):
        var value = Int(mapping[q])
        mapping_sum += value
        mapping_square_sum += value * value
    for row in range(count):
        var previous_sum = 0
        var previous_square_sum = 0
        for q in range(n_query):
            var value = Int(matches[row * n_query + q])
            previous_sum += value
            previous_square_sum += value * value
        if previous_sum != mapping_sum or previous_square_sum != mapping_square_sum:
            continue
        var same_set = True
        for q in range(n_query):
            var found = False
            for previous_q in range(n_query):
                if matches[row * n_query + previous_q] == mapping[q]:
                    found = True
                    break
            if not found:
                same_set = False
                break
        if same_set:
            return True
    return False


def candidate_ok(
    depth: Int,
    target: Int,
    mapping: I32Ptr,
    atom_compat: U8Ptr,
    query_u: I32Ptr,
    query_v: I32Ptr,
    parent_query: I32Ptr,
    parent_edge: I32Ptr,
    target_adjacency: I32Ptr,
    bond_compat: U8Ptr,
    n_target: Int,
    n_query_edges: Int,
    n_target_bonds: Int,
) -> Bool:
    if atom_compat[depth * n_target + target] == 0:
        return False
    var parent = Int(parent_query[depth])
    var checked_edge = -1
    if parent >= 0:
        checked_edge = Int(parent_edge[depth])
        var target_bond = Int(
            target_adjacency[target * n_target + Int(mapping[parent])]
        ) - 1
        if target_bond < 0:
            return False
        if bond_compat[checked_edge * n_target_bonds + target_bond] == 0:
            return False
    for edge in range(n_query_edges):
        if edge == checked_edge:
            continue
        var u = Int(query_u[edge])
        var v = Int(query_v[edge])
        var other = -1
        if u == depth and v < depth:
            other = v
        elif v == depth and u < depth:
            other = u
        if other >= 0:
            var target_bond = Int(
                target_adjacency[target * n_target + Int(mapping[other])]
            ) - 1
            if target_bond < 0:
                return False
            if bond_compat[edge * n_target_bonds + target_bond] == 0:
                return False
    return True


@export("mrd_substruct_matches")
def mrd_substruct_matches(
    atom_compat_addr: Int,
    query_u_addr: Int,
    query_v_addr: Int,
    parent_query_addr: Int,
    parent_edge_addr: Int,
    target_offsets_addr: Int,
    target_neighbors_addr: Int,
    target_adjacency_addr: Int,
    bond_compat_addr: Int,
    mapping_addr: Int,
    used_addr: Int,
    next_candidate_addr: Int,
    matches_addr: Int,
    n_query: Int,
    n_target: Int,
    n_query_edges: Int,
    n_target_bonds: Int,
    max_matches: Int,
    uniquify: Int,
) abi("C") -> Int:
    var atom_compat = U8Ptr(unsafe_from_address=atom_compat_addr)
    var query_u = I32Ptr(unsafe_from_address=query_u_addr)
    var query_v = I32Ptr(unsafe_from_address=query_v_addr)
    var parent_query = I32Ptr(unsafe_from_address=parent_query_addr)
    var parent_edge = I32Ptr(unsafe_from_address=parent_edge_addr)
    var target_offsets = I32Ptr(unsafe_from_address=target_offsets_addr)
    var target_neighbors = I32Ptr(unsafe_from_address=target_neighbors_addr)
    var target_adjacency = I32Ptr(unsafe_from_address=target_adjacency_addr)
    var bond_compat = U8Ptr(unsafe_from_address=bond_compat_addr)
    var mapping = I32Ptr(unsafe_from_address=mapping_addr)
    var used = U8Ptr(unsafe_from_address=used_addr)
    var next_candidate = I32Ptr(unsafe_from_address=next_candidate_addr)
    var matches = I32Ptr(unsafe_from_address=matches_addr)

    if n_query == 0 or n_query > n_target or max_matches <= 0:
        return 0
    for target in range(n_target):
        used[target] = UInt8(0)
    for q in range(n_query):
        next_candidate[q] = Int32(0)

    var depth = 0
    var count = 0
    while depth >= 0:
        var found = False
        var start = Int(next_candidate[depth])
        var candidate_count = n_target
        var neighbor_begin = 0
        var parent = Int(parent_query[depth])
        if parent >= 0:
            var anchor = Int(mapping[parent])
            neighbor_begin = Int(target_offsets[anchor])
            candidate_count = Int(target_offsets[anchor + 1]) - neighbor_begin
        for candidate in range(start, candidate_count):
            next_candidate[depth] = Int32(candidate + 1)
            var target = candidate
            if parent >= 0:
                target = Int(target_neighbors[neighbor_begin + candidate])
            if used[target] != 0:
                continue
            if candidate_ok(
                depth,
                target,
                mapping,
                atom_compat,
                query_u,
                query_v,
                parent_query,
                parent_edge,
                target_adjacency,
                bond_compat,
                n_target,
                n_query_edges,
                n_target_bonds,
            ):
                mapping[depth] = Int32(target)
                used[target] = UInt8(1)
                found = True
                if depth + 1 == n_query:
                    if uniquify == 0 or not contains_mapping(
                        matches, mapping, count, n_query
                    ):
                        for q in range(n_query):
                            matches[count * n_query + q] = mapping[q]
                        count += 1
                        if count == max_matches:
                            return count
                    used[target] = UInt8(0)
                else:
                    depth += 1
                    next_candidate[depth] = Int32(0)
                break
        if found:
            continue
        next_candidate[depth] = Int32(0)
        depth -= 1
        if depth >= 0:
            used[Int(mapping[depth])] = UInt8(0)
    return count
