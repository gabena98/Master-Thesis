const py = @import("python.zig");
const Self = @This();

array_type: ?*py.PyTypeObject,
from_any: *fn (?*py.PyObject, ?*Self.Descr, i64, i64, i64, ?*py.PyObject) ?*py.PyObject,
from_array: *fn (?*py.PyObject, ?*Self.Descr, i64) ?*py.PyObject,
descr_from_type: *fn (Self.Types) ?*Self.Descr,
new: *fn (?*py.PyTypeObject, c_int, [*]isize, Self.Types, ?[*]isize, ?*void, c_int, c_int, ?*py.PyObject) ?*py.PyObject,

pub fn from_api(api: [*]?*void) Self {
    return .{
        .array_type = @ptrCast(@alignCast(api[2])),
        .from_array = @ptrCast(@alignCast(api[109])),
        .from_any = @ptrCast(@alignCast(api[69])),
        .descr_from_type = @ptrCast(@alignCast(api[45])),
        .new = @ptrCast(@alignCast(api[93])),
    };
}

pub inline fn from_otf(self: Self, obj: ?*py.PyObject, _type: Self.Types, flags: i64) ?*py.PyObject {
    var descr = self.descr_from_type(_type);
    // @todo flags are actually mangled
    return self.from_any(obj, descr, 0, 0, flags, null);
}

pub inline fn simple_new(self: Self, nd: c_int, dims: [*]isize, typenum: Self.Types) ?*py.PyObject {
    return self.new(self.array_type, nd, dims, typenum, null, null, 0, 0, null);
}

pub const Array_Descr = extern struct {
    base: ?*Descr,
    shape: ?*py.PyObject,
};
pub const Array_Obj = extern struct {
    obj: py.PyObject,
    data: ?[*]u8,
    nd: c_int,
    dimensions: [*]isize,
    strides: [*]isize,
    base: ?*py.PyObject,
    descr: ?*Descr,
    flags: c_int,
    weakreflist: ?*py.PyObject,
    // @todo there may be other fields
};
pub const Descr = extern struct {
    base_ob: py.PyObject,
    typeobj: ?*py.PyTypeObject,
    kind: u8,
    type: u8,
    byteorder: u8,
    flags: u8,
    type_num: Types,
    elsize: i64,
    alignment: i64,
    _arr_descr: ?*Array_Descr,
    fields: py.PyObject,
    f: ?*void, // type would be `PyArray_ArrFuncs`
    metadata: ?*py.PyObject,
    c_metadata: ?*void, // type would be `NpyAuxData`
    hash: py.Py_hash_t,
};
pub const Types = enum(c_int) {
    BOOL = 0,
    BYTE,
    UBYTE,
    SHORT,
    USHORT,
    INT,
    UINT,
    LONG,
    ULONG,
    LONGLONG,
    ULONGLONG,
    FLOAT,
    DOUBLE,
    LONGDOUBLE,
    CFLOAT,
    CDOUBLE,
    CLONGDOUBLE,
    OBJECT = 17,
    STRING,
    UNICODE,
    VOID,
    DATETIME,
    TIMEDELTA,
    HALF,
    NTYPES,
    NOTYPE,
    USERDEF = 256,
    // NPY_NTYPES_ABI_COMPATIBLE = 21
};

pub const Array_Flags = extern struct {
    pub const C_CONTIGUOUS = @as(c_int, 0x0001);
    pub const F_CONTIGUOUS = @as(c_int, 0x0002);
    pub const OWNDATA = @as(c_int, 0x0004);
    pub const FORCECAST = @as(c_int, 0x0010);
    pub const ENSURECOPY = @as(c_int, 0x0020);
    pub const ENSUREARRAY = @as(c_int, 0x0040);
    pub const ELEMENTSTRIDES = @as(c_int, 0x0080);
    pub const ALIGNED = @as(c_int, 0x0100);
    pub const NOTSWAPPED = @as(c_int, 0x0200);
    pub const WRITEABLE = @as(c_int, 0x0400);
    pub const WRITEBACKIFCOPY = @as(c_int, 0x2000);
    pub const ENSURENOCOPY = @as(c_int, 0x4000);
    pub const BEHAVED = ALIGNED | WRITEABLE;
    pub const BEHAVED_NS = (ALIGNED | WRITEABLE) | NOTSWAPPED;
    pub const CARRAY = C_CONTIGUOUS | BEHAVED;
    pub const CARRAY_RO = C_CONTIGUOUS | ALIGNED;
    pub const FARRAY = F_CONTIGUOUS | BEHAVED;
    pub const FARRAY_RO = F_CONTIGUOUS | ALIGNED;
    pub const DEFAULT = CARRAY;
    pub const IN_ARRAY = CARRAY_RO;
    pub const OUT_ARRAY = CARRAY;
    pub const INOUT_ARRAY = CARRAY;
    pub const INOUT_ARRAY2 = CARRAY | WRITEBACKIFCOPY;
    pub const IN_FARRAY = FARRAY_RO;
    pub const OUT_FARRAY = FARRAY;
    pub const INOUT_FARRAY = FARRAY;
    pub const INOUT_FARRAY2 = FARRAY | WRITEBACKIFCOPY;
    pub const UPDATE_ALL = (C_CONTIGUOUS | F_CONTIGUOUS) | ALIGNED;
    pub const HAS_DESCR = @as(c_int, 0x0800);
};
