pub export fn __assert_fail(
    _assertion: [*:0]const u8,
    _file: [*:0]const u8,
    _line: u32,
    _function: [*:0]const u8,
) void {
    // Consuma i parametri per evitare errori
    _ = _assertion;
    _ = _file;
    _ = _line;
    _ = _function;

    @panic("assertion failed");
}

// This is chosen looking at the input table_c2c.
// Change this only if AoB errors occur (with the current ontology it doesnt)
const genealogy_max_size: usize = 12;

// number of parallel processes
const num_jobs: usize = 16;

// the numpy bindings
var np: Np = undefined; 

// Memory management
var _arena: std.heap.ArenaAllocator = undefined;
var global_arena: std.mem.Allocator = undefined;

// Python object to report errors. Refer to pyhton documentation
var generator_error: ?*py.PyObject = null;

// This python module will export three functions
// Refer to the C bindings documentation of python for the format
// 
// Functions referred to by this structure must be marked as `export` in their signature
const module_methods = [_]py.PyMethodDef{
    .{
        .ml_name = "create_c2c_table",
        .ml_meth = create_c2c_table,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Precompute the code2code distances",
    },
    .{
        .ml_name = "compute_patients_distances",
        .ml_meth = _compute_patients_distances,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "inputs a patient (ids + count), a list of patients (list of ids + list of counts) and a neighborhood size",
    },
    .{
        .ml_name = "ids_to_encoded",
        .ml_meth = _ids_to_encoded,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Perform the temporal encoding of doctorXAI",
    },
    .{
        .ml_name = "independent_perturbation",
        .ml_meth = _independent_perturbation,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Simple perturbation method. Unused",
    },
    .{
        .ml_name = "ontological_perturbation",
        .ml_meth = _ontological_perturbation,
        .ml_flags = py.METH_VARARGS,
        .ml_doc = "Ontological perturbation of doctorXAI",
    },
    // this one is just a sentinel
    // It is an array entry with all values set to zero
    std.mem.zeroes(py.PyMethodDef),
};

// precomputed table of code2code distances
var table_c2c: Table_c2c = undefined;
// Ontology
var ontology: []u32 = undefined;

/// inputs a patient (ids + count), a list of patients (list of ids + list of counts)
/// returns an an array of distances of each element of the list with the input patient
/// call this only after having filled the c2c table (called `create_c2c_table` from python)
export fn _compute_patients_distances(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var arg2: ?*py.PyObject = undefined;
    var arg3: ?*py.PyObject = undefined;
    var arg4: ?*py.PyObject = undefined;

    if (py.PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4) == 0) return null;

    const patient_codes = arg1;
    const patient_count = arg2;
    const dataset_codes = arg3;
    const dataset_count = arg4;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const patient = parse_patient(patient_codes, patient_count, allocator) orelse {
        py.PyErr_SetString(generator_error, "Error while parsing patient");
        return null;
    };

    const dataset = parse_list_of_patients(dataset_codes, dataset_count, allocator) orelse return null;

    var result_data: []f32 = undefined;
    const result_array = blk: {
        var dimensions = [_]isize{@intCast(dataset.len)};
        var obj = np.simple_new(dimensions.len, &dimensions, Np.Types.FLOAT) orelse {
            py.PyErr_SetString(generator_error, "Failed while creating result array");
            return null;
        };
        var arr: *Np.Array_Obj = @ptrCast(obj);
        result_data = @as([*]f32, @ptrCast(@alignCast(arr.data)))[0..dataset.len];
        break :blk obj;
    };

    compute_patients_distances(patient, dataset, table_c2c, result_data);

    return result_array;
}

/// This takes the list of patients to encode (ids and counts) and the max id, then returns their encoded form
/// with temporal encoding
export fn _ids_to_encoded(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var arg2: ?*py.PyObject = undefined;
    var arg3: i64 = undefined;
    var arg4: f32 = undefined;

    if (py.PyArg_ParseTuple(args, "OOlf", &arg1, &arg2, &arg3, &arg4) == 0) return null;

    const max_id: usize = @intCast(arg3);
    const lambda: f32 = arg4;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const dataset = parse_list_of_patients(arg1, arg2, allocator) orelse return null;

    var encoded_data: [*]f32 = undefined;
    const encoded_array = blk: {
        var dimensions = [_]isize{ @intCast(dataset.len), @intCast(max_id) };
        var obj = np.simple_new(dimensions.len, &dimensions, Np.Types.FLOAT) orelse {
            py.PyErr_SetString(generator_error, "Failed while creating `encoded` result array");
            return null;
        };
        var arr: *Np.Array_Obj = @ptrCast(obj);
        encoded_data = @ptrCast(@alignCast(arr.data));
        break :blk obj;
    };

    ids_to_encoded(dataset, encoded_data, @intCast(max_id), lambda);

    return encoded_array;
}

export fn _independent_perturbation(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var arg2: ?*py.PyObject = undefined;
    var arg3: c_int = undefined;
    var arg4: f32 = undefined;

    if (py.PyArg_ParseTuple(args, "OOlf", &arg1, &arg2, &arg3, &arg4) == 0) return null;

    const keep_prob: f32 = arg4;
    const multiply_factor: usize = @intCast(arg3);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const dataset = parse_list_of_patients(arg1, arg2, allocator) orelse return null;
    const result = independent_perturbation(dataset, multiply_factor, keep_prob);

    return result;
}

export fn _ontological_perturbation(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var arg2: ?*py.PyObject = undefined;
    var arg3: c_int = undefined;
    var arg4: f32 = undefined;

    if (py.PyArg_ParseTuple(args, "OOlf", &arg1, &arg2, &arg3, &arg4) == 0) return null;

    const keep_prob: f32 = arg4;
    const multiply_factor: usize = @intCast(arg3);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const dataset = parse_list_of_patients(arg1, arg2, allocator) orelse return null;
    const result = ontological_perturbation(dataset, multiply_factor, keep_prob, ontology);

    return result;
}
const Table_c2c = struct {
    table: []f32,
    w: usize,

    inline fn get(self: Table_c2c, c1: u32, c2: u32) f32 {
        return self.table[c1 * self.w + c2];
    }

    inline fn get_mut(self: Table_c2c, c1: u32, c2: u32) *f32 {
        return &self.table[c1 * self.w + c2];
    }
};

/// Precomputes the code2code distances
export fn create_c2c_table(self_obj: ?*py.PyObject, args: ?*py.PyObject) ?*py.PyObject {
    // Unwrap arguments to a plain array
    _ = self_obj;
    var arg1: ?*py.PyObject = undefined;
    var arg2: ?*py.PyObject = undefined;

    if (py.PyArg_ParseTuple(args, "OO", &arg1, &arg2) == 0) return null;
    const _arr = np.from_otf(arg1, Np.Types.UINT, Np.Array_Flags.IN_ARRAY) orelse return null;
    const arr: *Np.Array_Obj = @ptrCast(_arr);

    const _arr2 = np.from_otf(arg2, Np.Types.UINT, Np.Array_Flags.IN_ARRAY) orelse return null;
    const arr2: *Np.Array_Obj = @ptrCast(_arr2);

    var size: usize = undefined;
    var tree: []u32 = undefined;
    {
        const data: [*]u32 = @ptrCast(@alignCast(arr.data));
        const nd = arr.nd;
        if (nd != 2) {
            py.PyErr_SetString(generator_error, "Array should have a single dimension");
            return null;
        }
        const num_fields: usize = @intCast(arr.dimensions[1]);
        if (num_fields != 2) {
            const message = std.fmt.allocPrint(
                global_arena,
                "Array should have dim (*, 2), found dim ({}, {})",
                .{ arr.dimensions[0], arr.dimensions[1] },
            ) catch "Error";
            py.PyErr_SetString(generator_error, @ptrCast(message));
            return null;
        }
        size = @intCast(arr.dimensions[0]);
        tree = data[0 .. 2 * size];
    }

    const leaf_indices: []u32 = blk: {
        const t: [*]u32 = @ptrCast(@alignCast(arr2.data));
        const _size: usize = @intCast(arr2.dimensions[0]); // @todo check nd
        break :blk t[0.._size];
    };
    // end of unwrapping
    //
    const max_leaf_index = std.mem.max(u32, leaf_indices) + 1;
    const max_id = blk: {
        var max: u32 = 0;
        for (0..size) |it| {
            max = @max(tree[2 * it], max);
        }
        break :blk max;
    };

    // create and fill the ontology in a tree form
    var _ontology_tree = global_arena.alloc(u32, max_id + 1) catch return null;
    for (0..size) |it| {
        const child = tree[2 * it];
        const parent = tree[2 * it + 1];
        _ontology_tree[child] = parent;
    }

    // create and fill c2c table
    table_c2c = std.mem.zeroes(Table_c2c);
    table_c2c.table = global_arena.alloc(f32, max_leaf_index * max_leaf_index) catch @panic("Alloc error");
    table_c2c.w = max_leaf_index;
    @memset(table_c2c.table, std.math.nan(f32));

    // this if distinguishes between the single-threaded and the multi-threaded implementations
    // the result on the two branches should be the same
    if (comptime num_jobs == 1) {
        for (leaf_indices, 0..) |it, it_index| {
            for (leaf_indices[0 .. it_index + 1]) |jt| {
                const dist = compute_c2c(it, jt, _ontology_tree);
                table_c2c.get_mut(it, jt).* = dist; // method in the Table_c2c struct
                table_c2c.get_mut(jt, it).* = dist; // method in the Table_c2c struct
            }
        }
    } else {
        const Thread_Data = struct {
            ontology_tree: []u32,
            leaf_indices: []u32,
            table_c2c: Table_c2c,
            start: usize,
            len: usize,

            const Self = @This();

            pub fn job(self: Self) void {
                for (self.leaf_indices[self.start .. self.start + self.len], 0..) |it, it_index| {
                    for (self.leaf_indices[0 .. self.start + it_index + 1]) |jt| {
                        const dist = compute_c2c(it, jt, self.ontology_tree);
                        table_c2c.get_mut(it, jt).* = dist;
                        table_c2c.get_mut(jt, it).* = dist;
                    }
                }
            }
        };

        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const spawn_config = std.Thread.SpawnConfig{ .allocator = allocator };

        const base_size = leaf_indices.len / num_jobs;
        var remainder = leaf_indices.len % num_jobs;

        var threads = allocator.alloc(std.Thread, num_jobs) catch @panic("error in allocation");

        // @bug workload is not balanced on the threads
        var cursor: usize = 0;
        for (threads) |*thread| {
            var true_size = base_size;
            if (remainder > 0) {
                true_size += 1;
                remainder -= 1;
            }
            const thread_data = Thread_Data{
                .ontology_tree = _ontology_tree,
                .leaf_indices = leaf_indices,
                .table_c2c = table_c2c,
                .start = cursor,
                .len = true_size,
            };
            cursor += true_size;
            thread.* = std.Thread.spawn(spawn_config, Thread_Data.job, .{thread_data}) catch @panic("cannot create thread");
        }

        for (threads) |thread| {
            thread.join();
        }
    }

    ontology = _ontology_tree;

    // returns
    py.Py_INCREF(py.Py_None);
    return py.Py_None;
}

// Utility function to convert python datasets of patients in plain arrays
fn parse_list_of_patients(codes: ?*py.PyObject, counts: ?*py.PyObject, allocator: std.mem.Allocator) ?[][][]u32 {
    if (py.PyList_Check(codes) == 0) {
        py.PyErr_SetString(generator_error, "Argument `codes` should be a list");
        return null;
    }
    if (py.PyList_Check(counts) == 0) {
        py.PyErr_SetString(generator_error, "Argument `counts` should be a list");
        return null;
    }

    const list_len = blk: {
        const _list_len = py.PyList_GET_SIZE(codes);
        if (_list_len != py.PyList_GET_SIZE(counts)) {
            py.PyErr_SetString(generator_error, "Argument `codes` and `counts` should have the same size");
            return null;
        }
        break :blk @as(usize, @intCast(_list_len));
    };

    const dataset = blk: {
        var dataset = allocator.alloc([][]u32, list_len) catch return null;
        for (0..list_len) |it| {
            const codes_obj = py.PyList_GetItem(codes, @intCast(it));
            const count_obj = py.PyList_GetItem(counts, @intCast(it));
            //std.debug.print("Parsing patient {}...\n", .{it});
            dataset[it] = parse_patient(codes_obj, count_obj, allocator) orelse {
                const msg = std.fmt.allocPrintZ(global_arena, "Error while parsing dataset patient {}", .{it}) catch return null;
                py.PyErr_SetString(generator_error, msg);
                return null;
            };
        }
        break :blk dataset;
    };

    return dataset;
}

// Utility function to convert a patient from python object format to a plain array.
fn parse_patient(codes: ?*py.PyObject, counts: ?*py.PyObject, allocator: std.mem.Allocator) ?[][]u32 {
    const patient_id = blk: {
    const _patient_id = np.from_otf(codes, Np.Types.UINT, Np.Array_Flags.IN_ARRAY);
    if (_patient_id == null) {
        std.debug.print("Errore: impossibile convertire `codes` in array NumPy\n", .{});
        return null;
    }
    break :blk @as(*Np.Array_Obj, @ptrCast(_patient_id orelse return null));
    };
    const patient_cc = blk: {
        const _patient_cc = np.from_otf(counts, Np.Types.UINT, Np.Array_Flags.IN_ARRAY);
        if (_patient_cc == null) {
            std.debug.print("Errore: impossibile convertire `counts` in array NumPy\n", .{});
            return null;
        }
        break :blk @as(*Np.Array_Obj, @ptrCast(_patient_cc orelse return null));
    };

    if (patient_id.nd != 1) {
        std.debug.print("Errore: patient_id.nd ({}) non è 1\n", .{patient_id.nd});
        return null;
    }
    if (patient_cc.nd != 1) {
        std.debug.print("Errore: patient_cc.nd ({}) non è 1\n", .{patient_cc.nd});
        return null;
    }
    //std.debug.print("patient_cc.nd: {}, patient_cc.dimensions[0]: {}\n", .{patient_cc.nd, patient_cc.dimensions[0]});
    const num_visits: usize = @intCast(patient_cc.dimensions[0]);
    //std.debug.print("Allocazione memoria per {} visite...\n", .{num_visits});
    //std.debug.print("patient_id.nd: {}, patient_id.dimensions[0]: {}\n", .{patient_id.nd, patient_id.dimensions[0]});
    const data: [*]u32 = @ptrCast(@alignCast(patient_id.data));
    //std.debug.print("Dati patient_id: {any}\n", .{data[0..@intCast(patient_id.dimensions[0])]});
    var patient = allocator.alloc([]u32, num_visits) catch {
        std.debug.print("Errore: impossibile allocare memoria per {} visite\n", .{num_visits});
        return null;
    };
    const visit_lens: []u32 = @as([*]u32, @ptrCast(@alignCast(patient_cc.data)))[0..num_visits];
    //const data: [*]u32 = @ptrCast(@alignCast(patient_id.data));

    var cursor: usize = 0;
    //std.debug.print("visit_lens completo: {any}, num_visits: {}\n", .{visit_lens, num_visits});
    for (visit_lens, 0..) |c, it| {
        const len: usize = @intCast(c);
        //std.debug.print("Visit {}: lunghezza {}\n", .{it, len});
        patient[it] = data[cursor .. cursor + len];
        cursor += len;
        //std.debug.print("Cursor start: {}, Cursor end: {}\n", .{cursor, cursor + len});
    }
    const num_codes: usize = @intCast(patient_id.dimensions[0]);
    //std.debug.print("Numero totale di codici: {}, Cursor finale: {}\n", .{num_codes, cursor});
    if (cursor != num_codes) {
        std.debug.print("Errore: il numero totale di codici ({}) non corrisponde al cursor ({})\n", .{num_codes, cursor});
        return null;
    }
    return patient;
}

// compute the distance of the two codes using the ontology
fn compute_c2c(id_1: u32, id_2: u32, _ontology_tree: []u32) f32 {
    if (id_1 == id_2) return 0;

    const res_1 = get_genealogy(id_1, _ontology_tree);
    const res_2 = get_genealogy(id_2, _ontology_tree);

    const genealogy_1 = res_1[0];
    const genealogy_2 = res_2[0];
    const root_1 = res_1[1];
    const root_2 = res_2[1];

    var cursor_1 = root_1;
    var cursor_2 = root_2;
    while (genealogy_1[cursor_1] == genealogy_2[cursor_2]) {
        if (cursor_1 == 0 or cursor_2 == 0) break;
        cursor_1 -= 1;
        cursor_2 -= 1;
    }
    cursor_1 = @min(cursor_1 + 1, root_1);
    cursor_2 = @min(cursor_2 + 1, root_2);

    const d_lr_doubled: f32 = @floatFromInt(2 * (root_1 - cursor_1));
    const dist = 1.0 - d_lr_doubled / (@as(f32, @floatFromInt(cursor_1 + cursor_2)) + d_lr_doubled);

    return dist;
}

// utility function to follow the genealogy on the ontology tree
fn get_genealogy(id: u32, _ontology_tree: []u32) struct { [genealogy_max_size]u32, usize } {
    var res = std.mem.zeroes([genealogy_max_size]u32);
    res[0] = id;
    var it: usize = 0;
    while (true) {
        if (res[it] >= _ontology_tree.len) {
            break; // Oppure gestisci l'errore in altro modo
        }
        const parent = _ontology_tree[res[it]];
        if (parent != res[it]) {
            it += 1;
            res[it] = parent;
        } else break;
    }
    return .{ res, it };
}

// compute the asymmetrical visit2visit distance
fn asymmetrical_v2v(v1: []u32, v2: []u32, _table_c2c: Table_c2c) f32 {
    var sum: f32 = 0;
    for (v1) |c1| {
        var best = std.math.floatMax(f32);
        for (v2) |c2| {
            const dist = _table_c2c.get(c1, c2);
            best = @min(best, dist);
        }
        sum += best;
    }
    return sum;
}

// compute the visit2visit distance
fn compute_v2v(v1: []u32, v2: []u32, _table_c2c: Table_c2c) f32 {
    const x = asymmetrical_v2v(v1, v2, _table_c2c);
    const y = asymmetrical_v2v(v2, v1, _table_c2c);
    return @max(x, y);
}

fn compute_p2p(p1: [][]u32, p2: [][]u32, _table_c2c: Table_c2c, allocator: std.mem.Allocator) f32 {
    var table = allocator.alloc(f32, p1.len * p2.len) catch @panic("error with allocation");

    const w = p1.len;

    for (0..p1.len) |it| {
        for (0..p2.len) |jt| {
            const cost = compute_v2v(p1[it], p2[jt], _table_c2c);
            var in_cost: f32 = std.math.floatMax(f32);
            var del_cost: f32 = std.math.floatMax(f32);
            var edit_cost: f32 = std.math.floatMax(f32);

            if (it > 0) {
                in_cost = table[jt * w + it - 1];
                if (jt > 0) {
                    del_cost = table[(jt - 1) * w + it];
                    edit_cost = table[(jt - 1) * w + it - 1];
                }
            } else {
                if (jt > 0) {
                    del_cost = table[(jt - 1) * w + it];
                } else {
                    edit_cost = 0;
                }
            }

            table[jt * w + it] = cost + @min(in_cost, del_cost, edit_cost);
        }
    }

    return table[table.len - 1];
}

/// result is a preallocated array for the result of the same size of dataset
fn compute_patients_distances(patient: [][]u32, dataset: [][][]u32, _table_c2c: Table_c2c, result: []f32) void {
    if (comptime num_jobs == 1) {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        for (dataset, 0..) |d_patient, it| {
            const dist = compute_p2p(patient, d_patient, _table_c2c, allocator);
            result[it] = dist;
            _ = arena.reset(.retain_capacity);
        }
    } else {
        const Thread_Data = struct {
            table_c2c: Table_c2c,
            dataset: [][][]u32,
            patient: [][]u32,
            result: []f32,
            current_index: *usize,

            const Self = @This();
            const batch_size = 16;

            pub fn job(self: Self) void {
                var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                defer arena.deinit();
                const allocator = arena.allocator();

                while (true) {
                    const index: usize = @atomicRmw(usize, self.current_index, .Add, Self.batch_size, .Monotonic);
                    if (index >= self.dataset.len) break;

                    const index_limit = @min(self.dataset.len, index + batch_size);
                    for (self.dataset[index..index_limit], index..) |d_patient, it| {
                        const dist = compute_p2p(self.patient, d_patient, self.table_c2c, allocator);
                        self.result[it] = dist;
                        _ = arena.reset(.retain_capacity);
                    }
                }
            }
        };

        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const spawn_config = std.Thread.SpawnConfig{ .allocator = allocator };
        var threads = allocator.alloc(std.Thread, num_jobs) catch @panic("error in allocation");

        var current_index: usize = 0;

        for (threads) |*thread| {
            const thread_data = Thread_Data{
                .table_c2c = _table_c2c,
                .dataset = dataset,
                .patient = patient,
                .result = result,
                .current_index = &current_index,
            };
            thread.* = std.Thread.spawn(spawn_config, Thread_Data.job, .{thread_data}) catch @panic("cannot create thread");
        }

        for (threads) |thread| {
            thread.join();
        }
    }
}

/// perform temporal encodings on the dataset
fn ids_to_encoded(dataset: [][][]u32, encoded: [*]f32, max_label: u32, lambda: f32) void {
    const data_size = dataset.len;
    const index = Indexer(2).init(.{ data_size, max_label });

    @memset(encoded[0..index.size], 0);

    for (dataset, 0..) |patient, it| {
        var factor: f32 = 0.5;
        for (0..patient.len) |jt| {
            defer factor *= lambda;

            const visit_it = patient.len - jt - 1;
            for (patient[visit_it]) |c| {
                encoded[index.ix(.{ it, c })] += factor;
            }
        }

    }
}

// simple perturbation. Unused
fn independent_perturbation(dataset: [][][]u32, multiply_factor: usize, keep_prob: f32) *py.PyObject {
    var ids_list = py.PyList_New(@intCast(dataset.len * multiply_factor));
    var counts_list = py.PyList_New(@intCast(dataset.len * multiply_factor));

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var _rng = std.rand.DefaultPrng.init(42);
    var rng = _rng.random();

    for (dataset, 0..) |patient, it| {
        for (0..multiply_factor) |jt| {
            const new_patient = independent_perturb_patient(patient, keep_prob, allocator, rng);
            const list_id: isize = @intCast(multiply_factor * it + jt);
            py.PyList_SET_ITEM(ids_list, list_id, new_patient.ids);
            py.PyList_SET_ITEM(counts_list, list_id, new_patient.counts);
            _ = arena.reset(.retain_capacity);
        }
    }

    const return_tuple = blk: {
        var tuple = py.PyTuple_New(2);
        _ = py.PyTuple_SetItem(tuple, 0, ids_list);
        _ = py.PyTuple_SetItem(tuple, 1, counts_list);
        break :blk tuple;
    };
    return return_tuple;
}

// Ontological perturbation
fn ontological_perturbation(dataset: [][][]u32, multiply_factor: usize, keep_prob: f32, _ontology: []u32) *py.PyObject {
    var ids_list = py.PyList_New(@intCast(dataset.len * multiply_factor));
    var counts_list = py.PyList_New(@intCast(dataset.len * multiply_factor));

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var _rng = std.rand.DefaultPrng.init(42);
    var rng = _rng.random();

    for (dataset, 0..) |patient, it| {
        for (0..multiply_factor) |jt| {
            const new_patient = ontological_perturb_patient(patient, keep_prob, _ontology, allocator, rng);
            const list_id: isize = @intCast(multiply_factor * it + jt);
            py.PyList_SET_ITEM(ids_list, list_id, new_patient.ids);
            py.PyList_SET_ITEM(counts_list, list_id, new_patient.counts);
            _ = arena.reset(.retain_capacity);
        }
    }

    const return_tuple = blk: {
        var tuple = py.PyTuple_New(2);
        _ = py.PyTuple_SetItem(tuple, 0, ids_list);
        _ = py.PyTuple_SetItem(tuple, 1, counts_list);
        break :blk tuple;
    };
    return return_tuple;
}

const Numpy_Patient = struct {
    ids: *py.PyObject,
    counts: *py.PyObject,
};

fn independent_perturb_patient(patient: [][]u32, keep_prob: f32, allocator: std.mem.Allocator, rng: std.rand.Random) Numpy_Patient {
    const new_visits = allocator.alloc([]bool, patient.len) catch unreachable;
    var num_taken: usize = 0;
    for (patient, new_visits) |visit, *new_visit| {
        if (visit.len == 0) @panic("we cannot have empty visits in input");
        new_visit.* = allocator.alloc(bool, visit.len) catch unreachable;
        var num_visit_taken: usize = 0;
        for (new_visit.*) |*it| {
            const r = rng.float(f32);
            it.* = r < keep_prob;
            if (it.*) num_visit_taken += 1;
        }
        if (num_visit_taken == 0) {
            num_visit_taken = 1;
            new_visit.*[0] = true;
        }
        num_taken += num_visit_taken;
    }

    const ids = np.simple_new(1, @ptrCast(&num_taken), Np.Types.UINT) orelse unreachable;
    const counts = np.simple_new(1, @constCast(@ptrCast(&patient.len)), Np.Types.UINT) orelse unreachable;
    const ids_data: [*]u32 = @ptrCast(@alignCast(@as(*Np.Array_Obj, @ptrCast(ids)).data));
    const counts_data: [*]u32 = @ptrCast(@alignCast(@as(*Np.Array_Obj, @ptrCast(counts)).data));

    var cursor: usize = 0;
    var source_cursor: u32 = 0;
    for (new_visits, counts_data) |new_visit, *count| {
        var counter: u32 = 0;
        for (new_visit) |taken| {
            if (taken) {
                ids_data[cursor] = source_cursor;
                cursor  += 1;
                counter += 1;
            }
            source_cursor += 1;
        }
        count.* = counter;
    }

    return Numpy_Patient{
        .ids = ids,
        .counts = counts,
    };
}

fn is_in(array: []u32, val: u32) bool {
    for (array) |it| {
        if (it == val) return true;
    }
    return false;
}

fn ontological_perturb_patient(patient: [][]u32, keep_prob: f32, _ontology: []u32, allocator: std.mem.Allocator, rng: std.rand.Random) Numpy_Patient {
    const new_visits = allocator.alloc([]bool, patient.len) catch unreachable;
    var num_taken: usize = 0;

    var masked_parents = std.ArrayList(u32).initCapacity(allocator, patient.len) catch unreachable;

    for (patient, new_visits) |visit, *new_visit| {
        if (visit.len == 0) @panic("we cannot have empty visits in input");
        new_visit.* = allocator.alloc(bool, visit.len) catch unreachable;

        var num_visit_taken: usize = 0;

        for (visit, new_visit.*) |c, *it| {
            const p = _ontology[c];
            if (is_in(masked_parents.items, p)) {
                it.* = false;
            } else {
                const r = rng.float(f32);
                it.* = r < keep_prob;
                if (it.*) {
                    num_visit_taken += 1;
                }
                else masked_parents.append(p) catch unreachable;
            }
        }

        // @todo instead of taking an empty visit, we default to taking the first code
        // This should *almost* never happen
        if (num_visit_taken == 0) {
            num_visit_taken = 1;
            new_visit.*[0] = true;
        }
        num_taken += num_visit_taken;
    }

    const ids = np.simple_new(1, @ptrCast(&num_taken), Np.Types.UINT) orelse unreachable;
    const counts = np.simple_new(1, @constCast(@ptrCast(&patient.len)), Np.Types.UINT) orelse unreachable;
    const ids_data: [*]u32 = @ptrCast(@alignCast(@as(*Np.Array_Obj, @ptrCast(ids)).data));
    const counts_data: [*]u32 = @ptrCast(@alignCast(@as(*Np.Array_Obj, @ptrCast(counts)).data));

    var cursor: usize = 0;
    var source_cursor: u32 = 0;
    for (new_visits, counts_data) |new_visit, *count| {
        var counter: u32 = 0;
        for (new_visit) |taken| {
            if (taken) {
                ids_data[cursor] = source_cursor;
                cursor  += 1;
                counter += 1;
            }
            source_cursor += 1;
        }
        count.* = counter;
    }

    return Numpy_Patient{
        .ids = ids,
        .counts = counts,
    };
}

// Required for python bindings
var generator_module = py.PyModuleDef{
    .m_base = .{
        .ob_base = .{ .ob_refcnt = 1, .ob_type = null },
        .m_init = null,
        .m_index = 0,
        .m_copy = null,
    },
    .m_name = "generator",
    .m_doc = "generator module",
    .m_size = -1,
    .m_methods = @constCast(&module_methods),
    .m_slots = null,
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

// Required for python bindings, entry point of the module
// Imports numpy too
pub export fn PyInit_generator() ?*py.PyObject {
    const module = py.PyModule_Create(@constCast(&generator_module));
    if (module == null) return null;

    generator_error = py.PyErr_NewException("generator.error", null, null);
    py.Py_XINCREF(generator_error);
    if (py.PyModule_AddObject(module, "error", generator_error) < 0) {
        py.Py_XDECREF(generator_error);
        {
            const tmp = generator_error;
            if (tmp != null) {
                generator_error = null;
                py.Py_DECREF(tmp);
            }
        }
        py.Py_DECREF(module);
        return null;
    }

    np = import_numpy() catch std.debug.panic("cannot import numpy", .{});

    _arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    global_arena = _arena.allocator();

    return module;
}

fn import_numpy() !Np {
    // @todo this can fail in so many ways... See `__multiarray_api.h` line 1483
    const numpy = py.PyImport_ImportModule("numpy.core._multiarray_umath");
    if (numpy == null) return error.generic;
    const c_api = py.PyObject_GetAttrString(numpy, "_ARRAY_API");
    if (c_api == null) return error.generic;
    const PyArray_api = blk: {
        const t = py.PyCapsule_GetPointer(c_api, null);
        if (t == null) return error.generic;
        const ret: [*]?*void = @ptrCast(@alignCast(t));
        break :blk ret;
    };
    return Np.from_api(PyArray_api);
}

const std = @import("std");
const py = @import("python.zig");

const Np = @import("numpy_data.zig");
const Indexer = @import("tensor.zig").Indexer;
