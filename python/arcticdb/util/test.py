"""
Copyright 2023 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file licenses/BSL.txt.

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
"""
import copy
import os
from contextlib import contextmanager
from typing import Mapping, Any, Optional, NamedTuple, List, AnyStr, Union, Dict
import numpy as np
import pandas as pd
from pandas import DateOffset, Timedelta
from pandas._typing import Scalar
import datetime as dt
import string
import random
import time
import attr
from functools import wraps, reduce

from arcticdb.util.marks import SHORTER_LOGS

try:
    from pandas.errors import UndefinedVariableError
except ImportError:
    from pandas.core.computation.ops import UndefinedVariableError

from arcticdb import QueryBuilder
from arcticdb.util._versions import IS_PANDAS_ONE, PANDAS_VERSION, CHECK_FREQ_VERSION
from arcticdb.version_store import NativeVersionStore
from arcticdb.version_store._custom_normalizers import CustomNormalizer
from arcticc.pb2.descriptors_pb2 import NormalizationMetadata
from arcticc.pb2.storage_pb2 import LibraryDescriptor, VersionStoreConfig
from arcticdb.version_store.helper import ArcticFileConfig
from arcticdb.config import _DEFAULT_ENVS_PATH
from arcticdb_ext import (
    set_config_int,
    get_config_int,
    unset_config_int,
    get_config_string,
    set_config_string,
    unset_config_string,
)
from packaging.version import Version

def create_df(start=0, columns=1) -> pd.DataFrame:
    data = {}
    for i in range(columns):
        col_name = chr(ord("x") + i)  # Generates column names like 'x', 'y', 'z', etc.
        data[col_name] = np.arange(start + i * 10, start + (i + 1) * 10, dtype=np.int64)

    index = np.arange(start, start + 10, dtype=np.int64)
    return pd.DataFrame(data, index=index)


def create_df_index_rownum(num_columns: int, start_index: int, end_index: int) -> pd.DataFrame:
    """
    Creates a data frame with (integer index) specified number of columns with integer index starting
    from specified and ending in specified position. The content of the dataframe is
    integer random numbers ['start_index', 'end_index')

    Why this is useful? Consider this example:

      df1 = create_df_index_rownum(num_columns=2, start_index=0, end_index=2)
      df2 = create_df_index_rownum(num_columns=2, start_index=2, end_index=4)
      df_full = pd.concat([df1, df2])

    This will produce 3 dataframes and the result of the update full operation would be
    visibly correct, even without other methods of comparison:

        df1:
        COL_0  COL_1
        0      1      1
        1      1      0
        --------------------
        df2:
        COL_0  COL_1
        2      3      2
        3      2      2
        --------------------
        df_full:
        COL_0  COL_1
        0      1      1
        1      1      0
        2      3      2
        3      2      2

    """
    rows = end_index - start_index
    cols = [f"COL_{i}" for i in range(num_columns)]
    rng = np.random.default_rng()
    df = pd.DataFrame(rng.integers(start_index, end_index, size=(rows, num_columns), dtype=np.int64), columns=cols)
    df.index = np.arange(start_index, end_index, 1)
    return df


def create_datetime_index(
    df: pd.DataFrame,
    name_col: str,
    freq: Union[str, dt.timedelta, Timedelta, DateOffset],
    start_time: pd.Timestamp = pd.Timestamp(0),
):
    """
    creates a datetime index to a dataframe. The index will start at specified timestamp
    and will be generated using specified frequency having same number of periods as the rows
    in the table
    """
    periods = len(df)
    index = pd.date_range(start=start_time, periods=periods, freq=freq, name=name_col)
    df.index = index


def create_df_index_datetime(num_columns: int, start_hour: int, end_hour: int) -> pd.DataFrame:
    """
    Creates data frame with specified number of columns
    with datetime index sorted, starting from start_hour till
    end hour (you can use thousands of hours if needed).
    The data is random integer data again with min 'start_hour' and
    max 'end_hour'. The goal is to achieve same visually understandable
    correctness validation of append/update operations (see 'create_df_index_rownum()' doc)
    """
    assert start_hour >= 0, "start index must be positive"
    assert end_hour >= 0, "end index must be positive"

    time_start = dt.datetime(2000, 1, 1, 0)
    rows = end_hour - start_hour
    cols = [f"COL_{i}" for i in range(num_columns)]
    rng = np.random.default_rng()
    df = pd.DataFrame(rng.integers(start_hour, end_hour, size=(rows, num_columns), dtype=np.int64), columns=cols)
    start_date = time_start + dt.timedelta(hours=start_hour)
    dr = pd.date_range(start_date, periods=rows, freq="h").astype("datetime64[ns]")
    df.index = dr
    return df


def dataframe_dump_to_log(label_for_df, df: pd.DataFrame):
    """
    Useful for printing in log content and data types of columns of
    a dataframe. This way in the full log of a test we will have actual dataframe, that later
    we could use to reproduce something or further analyze failures caused by
    a problems in test code or arctic
    """
    print("-" * 80)
    if SHORTER_LOGS:
        print("No full dataframes dumps when shorter logs is activated")
    else:
        if isinstance(df, pd.DataFrame):
            print("dataframe : , ", label_for_df)
            print(df.to_csv())
            print("column definitions : ")
            print(df.dtypes)
        else:
            print(f"Not a dataframe passed : {type(df)}")
            print(df)
        print("-" * 80)


def dataframe_simulate_arcticdb_update_static(existing_df: pd.DataFrame, update_df: pd.DataFrame) -> pd.DataFrame:
    """
    Does implement arctic logic of update() method functionality over pandas dataframes.
    In other words the result, new data frame will have the content of 'existing_df' dataframe
    updated with the content of "update_df" dataframe the same way that arctic is supposed to work.
    Useful for prediction of result content of arctic database after update operation
    NOTE: you have to pass indexed dataframe
    """

    assert existing_df.dtypes.to_list() == update_df.dtypes.to_list(), (
        "Dataframe must have identical columns types in same order"
    )
    assert existing_df.columns.to_list() == update_df.columns.to_list(), "Columns names also need to be in same order"

    start2 = update_df.first_valid_index()
    end2 = update_df.last_valid_index()

    chunks = []
    df1 = existing_df[existing_df.index < start2]
    chunks.append(df1)
    chunks.append(update_df)
    df2 = existing_df[existing_df.index > end2]
    chunks.append(df2)
    result_df = pd.concat(chunks)
    return result_df


def dataframe_single_column_string(length=1000, column_label="string_short", seed=0, string_len=1):
    """
    creates a dataframe with one column, which label can be changed, containing string
    with specified length. Useful for combining this dataframe with another dataframe
    """
    np.random.seed(seed)
    return pd.DataFrame({column_label: [random_string(string_len) for _ in range(length)]})


def dataframe_filter_with_datetime_index(
    df: pd.DataFrame, start_timestamp: Scalar, end_timestamp: Scalar, inclusive="both"
) -> pd.DataFrame:
    """
    Filters dataframe which has datetime index, and selects dates from start till end,
    where inclusive can be one of (both,left,right,neither)
    start and end can be pandas.Timeframe, datetime or string datetime
    """

    return df[df.index.to_series().between(start_timestamp, end_timestamp, inclusive="both")]


def maybe_not_check_freq(f):
    """Ignore frequency when pandas is newer as starts to check frequency which it did not previously do."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if PANDAS_VERSION >= CHECK_FREQ_VERSION and "check_freq" not in kwargs:
            kwargs["check_freq"] = False
        try:
            return f(*args, **kwargs)
        except AssertionError as ae:
            df = []
            if "left" in kwargs:
                df = kwargs["left"]
            else:
                df = args[0]
            dataframe_dump_to_log("LEFT dataframe (expected)", df)
            if "right" in kwargs:
                df = kwargs["right"]
            else:
                df = args[1]
            dataframe_dump_to_log("RIGHT dataframe (actual)", df)
            raise ae

    return wrapper


assert_frame_equal = maybe_not_check_freq(pd.testing.assert_frame_equal)
assert_series_equal = maybe_not_check_freq(pd.testing.assert_series_equal)


def assert_frame_equal_rebuild_index_first(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """
    Use for dataframes that have index row range and you
    obtain data from arctic with QueryBuilder.

    First will rebuild index for dataframes to assure we
    have same index in both frames when row range index is used
    """
    if IS_PANDAS_ONE:
        # On zero size frames some column types will not match when pandas 1.x is used
        if expected.shape[0] == expected.shape[0] and expected.shape[0] == 0:
            assert_frame_equal(left=expected, right=actual, check_dtype=False)
    expected.reset_index(inplace=True, drop=True)
    actual.reset_index(inplace=True, drop=True)
    assert_frame_equal(left=expected, right=actual)


unicode_symbol = "\u00A0"  # start of latin extensions
unicode_symbols = "".join([chr(ord(unicode_symbol) + i) for i in range(100)])


def random_string(length: int):
    if random.randint(0, 3) == 0:
        # (probably) Give a unicode string one time in three, we have special handling in C++ for unicode
        return "".join(random.choice(string.ascii_uppercase + unicode_symbols) for _ in range(length))
    else:
        return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))


def get_sample_dataframe(size=1000, seed=0, str_size=10):
    np.random.seed(seed)
    df = pd.DataFrame(
        {
            "uint8": random_integers(size, np.uint8),
            "strings": [random_string(str_size) for _ in range(size)],
            "uint16": random_integers(size, np.uint16),
            "uint32": random_integers(size, np.uint32),
            "uint64": random_integers(size, np.uint64),
            "int8": random_integers(size, np.int8),
            "int16": random_integers(size, np.int16),
            "int32": random_integers(size, np.int32),
            "int64": random_integers(size, np.int64),
            "float32": np.random.randn(size).astype(np.float32),
            "float64": np.arange(size, dtype=np.float64),
            "bool": np.random.randn(size) > 0,
        }
    )
    return df


def get_sample_dataframe_only_strings(size=1000, seed=0, num_cols=1):
    np.random.seed(seed)
    df = pd.DataFrame({"strings" + str(idx): [random_string(10) for _ in range(size)] for idx in range(num_cols)})
    return df


def get_sample_dataframe_no_strings(size=1000, seed=0):
    np.random.seed(seed)
    df = pd.DataFrame(
        {
            "uint8": random_integers(size, np.uint8),
            "uint16": random_integers(size, np.uint16),
            "uint32": random_integers(size, np.uint32),
            "uint64": random_integers(size, np.uint64),
            "int8": random_integers(size, np.int8),
            "int16": random_integers(size, np.int16),
            "int32": random_integers(size, np.int32),
            "int64": random_integers(size, np.int64),
            "float32": np.random.randn(size).astype(np.float32),
            "float64": np.arange(size, dtype=np.float64),
            "bool": np.random.randn(size) > 0,
        }
    )
    return df


def get_lib_by_name(lib_name, env, conf_path=_DEFAULT_ENVS_PATH):
    local_conf = ArcticFileConfig(env, conf_path)
    lib = local_conf[lib_name]
    return lib


@contextmanager
def config_context(name: str, value: int):
    initial = get_config_int(name)
    try:
        set_config_int(name, value)
        yield
    finally:
        if initial is not None:
            set_config_int(name, initial)
        else:
            unset_config_int(name)


@contextmanager
def config_context_multi(config: Dict[str, int]):
    """Call set_config_int for each entry in the dict."""
    initial = dict()
    try:
        for name, value in config.items():
            initial[name] = get_config_int(name)
            if value is None:
                unset_config_int(name)
            else:
                set_config_int(name, value)
        yield
    finally:
        for name, value in config.items():
            if name in initial and initial[name]:
                set_config_int(name, initial[name])
            else:
                unset_config_int(name)


@contextmanager
def config_context_string(name: str, value: str):
    try:
        initial = get_config_string(name)
        set_config_string(name, value)
        yield
    finally:
        if initial is not None:
            set_config_string(name, initial)
        else:
            unset_config_string(name)


CustomThing = NamedTuple(
    "CustomThing",
    [
        ("custom_index", np.ndarray),
        ("custom_columns", List[AnyStr]),
        ("custom_values", List[np.ndarray]),
    ],
)


class TestCustomNormalizer(CustomNormalizer):
    def normalize(self, item, **kwargs):
        if isinstance(item, CustomThing):
            norm_meta = NormalizationMetadata.CustomNormalizerMeta()
            df = pd.DataFrame(index=item.custom_index, columns=item.custom_columns, data=item.custom_values)
            return df, norm_meta

    def denormalize(self, item: Any, norm_meta: NormalizationMetadata.CustomNormalizerMeta) -> Any:
        return CustomThing(custom_index=item.index, custom_columns=item.columns, custom_values=item.values)

class CustomDict(dict):
    pass

class CustomDictNormalizer(CustomNormalizer):
    NESTED_STRUCTURE = True

    def normalize(self, item, **kwargs):
        if not isinstance(item, CustomDict):
            return None
        return dict(item), NormalizationMetadata.CustomNormalizerMeta()

    def denormalize(self, item, norm_meta):
        return CustomDict(item)

def sample_dataframe(size=1000, seed=0):
    return get_sample_dataframe(size, seed)


def sample_dataframe_only_strings(size=1000, seed=0, num_cols=1):
    return get_sample_dataframe_only_strings(size, seed, num_cols)


def sample_dataframe_without_strings(size=1000, seed=0):
    return get_sample_dataframe_no_strings(size, seed)


def populate_db(version_store):
    df = sample_dataframe()
    version_store.write("symbol", df)
    version_store.write("pickled", {"a": 1}, pickle_on_failure=True)
    version_store.snapshot("mysnap")
    version_store.write("rec_norm", data={"a": np.arange(5), "b": np.arange(8), "c": None}, recursive_normalizers=True)


def random_integers(size, dtype, min_value: int = None, max_value: int = None):
    # We do not generate integers outside the int64 range
    iinfo = np.iinfo(dtype)
    platform_int_info = iinfo
    if min_value is None:
        min_value = max(iinfo.min, platform_int_info.min)
    if max_value is None:
        max_value = min(iinfo.max, platform_int_info.max)
    return np.random.randint(min_value, max_value, size=size, dtype=dtype)


def get_wide_dataframe(size=10000, seed=0):
    np.random.seed(seed)
    return pd.DataFrame(
        {
            "uint8": random_integers(size, np.uint8),
            "strings": [random_string(100) for _ in range(size)],
            "uint16": random_integers(size, np.uint16),
            "uint32": random_integers(size, np.uint32),
            "uint64": random_integers(size, np.uint64),
            "int8": random_integers(size, np.int8),
            "int16": random_integers(size, np.int16),
            "int32": random_integers(size, np.int32),
            "int64": random_integers(size, np.int64),
            "float32": np.random.randn(size).astype(np.float32),
            "float64": np.arange(size, dtype=np.float64),
            "bool": np.random.randn(size) > 0,
        }
    )

def get_pickle():
    return (
        list(random_integers(10000, np.uint32)),
        str(random_string(100)),
        {"a": list(random_integers(100000, np.int32))},
    )[np.random.randint(0, 2)]


def random_ascii_strings(count, max_length):
    result = []
    for _ in range(count):
        length = random.randrange(max_length + 1)
        result.append("".join(random.choice(string.ascii_letters) for _ in range(length)))
    return result


def random_strings_of_length(num, length, unique=False):
    out = []
    for i in range(num):
        out.append(random_string(length))

    if unique:
        return list(set(out))
    else:
        return out


def random_strings_of_length_with_nan(num, length):
    out = []
    for i in range(num):
        if i % 3 == 1:
            out.append(np.nan)
        else:
            out.append("".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length)))

    return out


def random_floats(num):
    return np.random.uniform(low=0.5, high=20.0, size=(num,))


def random_dates(num):
    base_date = np.datetime64("2017-01-01")
    return np.array([base_date + random.randint(0, 100) for _ in range(num)])


def dataframe_for_date(dt, identifiers):
    length = len(identifiers)
    index = pd.MultiIndex.from_arrays([[dt] * length, identifiers])
    return pd.DataFrame(random_floats(length), index=index)


def get_symbols(lib, all_symbols=True):
    return sorted(lib.list_symbols(all_symbols=all_symbols))


def get_versions(lib, all_symbols=True):
    symbols = get_symbols(lib, all_symbols)
    versions_dict = dict()
    for sym in symbols:
        versions_list = lib.list_versions(sym)
        symbol_versions_dict = dict()
        for version in versions_list:
            symbol_versions_dict[version["version"]] = {k: version[k] for k in ["deleted", "snapshots"]}
        versions_dict[sym] = symbol_versions_dict
    return versions_dict


def get_snapshots(lib):
    return lib.list_snapshots()


def get_snapshot_versions(lib):
    snapshots = get_snapshots(lib)
    versions_dict = dict()
    for snapshot in snapshots.keys():
        snapshot_versions_list = lib.list_versions(snapshot=snapshot)
        snapshot_versions_dict = dict()
        for snapshot_version in snapshot_versions_list:
            snapshot_versions_dict[snapshot_version["symbol"]] = {
                k: snapshot_version[k] for k in ["version", "snapshots"]
            }
        versions_dict[snapshot] = snapshot_versions_dict
    return versions_dict


_WriteOpts = VersionStoreConfig.WriteOptions.DESCRIPTOR


def _calc_write_opt_one_ofs():
    for oneof in _WriteOpts.oneofs:
        for option in oneof.fields:
            if set(option.message_type.fields_by_name.keys()) == {"enabled"}:
                yield option.name


_WRITE_OPTION_ONE_OFS = tuple(_calc_write_opt_one_ofs())


def apply_lib_cfg(lib_cfg: LibraryDescriptor, cfg_dict: Mapping[str, Any]):
    """Used by library factory functions to apply configuration supplied as kwargs"""
    write_opts = lib_cfg.version.write_options
    for k, v in cfg_dict.items():
        if k in _WRITE_OPTION_ONE_OFS:
            setattr(getattr(write_opts, k), "enabled", v)
        elif k in _WriteOpts.fields_by_name:
            setattr(write_opts, k, v)
        elif k in VersionStoreConfig.DESCRIPTOR.fields_by_name:
            setattr(lib_cfg.version, k, v)
        elif k in VersionStoreConfig.MsgPack.DESCRIPTOR.fields_by_name:
            setattr(lib_cfg.version.msg_pack, k, v)
        elif k in LibraryDescriptor.DESCRIPTOR.fields_by_name:
            setattr(lib_cfg, k, v)
        else:
            raise NotImplementedError(k + " is not in the protobuf definitions")


def compare_version_data(source_lib, target_libs, versions):
    for symbol, symbol_versions in versions.items():
        for version, version_info in symbol_versions.items():
            source_vit = source_lib.read(symbol, as_of=version)
            for target_lib in target_libs:
                target_vit = target_lib.read(symbol, as_of=version)
                try:
                    compare_data(
                        source_vit.data,
                        source_vit.metadata,
                        target_vit.data,
                        target_vit.metadata,
                    )
                except AssertionError as e:
                    print("Version {} of symbol {} differs".format(version, symbol))
                    print("Source:\n{}".format(source_vit.data))
                    print("Target:\n{}".format(target_vit.data))
                    raise e


def compare_snapshot_data(source_lib, target_libs, snapshot_versions):
    for snapshot, symbol_versions in snapshot_versions.items():
        for symbol, version_info in symbol_versions.items():
            source_vit = source_lib.read(symbol, as_of=snapshot)
            for target_lib in target_libs:
                target_vit = target_lib.read(symbol, as_of=snapshot)
                try:
                    compare_data(
                        source_vit.data,
                        source_vit.metadata,
                        target_vit.data,
                        target_vit.metadata,
                    )
                except AssertionError as e:
                    print("Snapshot {} of symbol {} differs".format(snapshot, symbol))
                    print("Source:\n{}".format(source_vit.data))
                    print("Target:\n{}".format(target_vit.data))
                    raise e


def compare_data(source_data, source_metadata, target_data, target_metadata):
    if isinstance(source_data, pd.DataFrame):
        assert_frame_equal(source_data, target_data)
    else:
        # Recursive normalised symbol is a tuple of ndarrays
        assert len(source_data) == len(target_data)
        for idx in range(len(source_data)):
            assert np.allclose(source_data[idx], target_data[idx])
    assert source_metadata == target_metadata


def libraries_identical(source_lib, target_libs):
    # Assume if target_libs is not a list then it is a single library
    if not isinstance(target_libs, list):
        target_libs = [target_libs]
    with config_context("VersionMap.ReloadInterval", 0):
        symbols_source = get_symbols(source_lib)
        versions_source = get_versions(source_lib)
        snapshots_source = get_snapshots(source_lib)
        snapshot_versions_source = get_snapshot_versions(source_lib)

        for target_lib in target_libs:
            symbols_target = get_symbols(target_lib)
            if symbols_source != symbols_target:
                print("symbols_source != symbols_target")
                print("symbols_source {} : {}".format(source_lib, symbols_source))
                print("symbols_target {} : {}".format(target_lib, symbols_target))
                return False

            versions_target = get_versions(target_lib)
            if versions_source != versions_target:
                print("versions_source != versions_target")
                print("versions_source {} : {}".format(source_lib, versions_source))
                print("versions_target {} : {}".format(target_lib, versions_target))
                return False

            snapshots_target = get_snapshots(target_lib)
            if snapshots_source != snapshots_target:
                print("snapshots_source != snapshots_target")
                print("snapshots_source {} : {}".format(source_lib, snapshots_source))
                print("snapshots_target {} : {}".format(target_lib, snapshots_target))
                return False

            snapshot_versions_target = get_snapshot_versions(target_lib)
            if snapshot_versions_source != snapshot_versions_target:
                print("snapshot_versions_source != snapshots_target")
                print("snapshot_versions_source {} : {}".format(source_lib, snapshot_versions_source))
                print("snapshot_versions_target {} : {}".format(target_lib, snapshot_versions_target))
                return False

        compare_version_data(source_lib, target_libs, versions_source)
        compare_snapshot_data(source_lib, target_libs, snapshot_versions_source)
        return True


def make_dynamic(df, num_slices=10):
    cols = df.columns
    num_cols = len(cols)
    num_rows = len(df)
    rows_per_slice = int(num_rows / num_slices)
    rows_per_slice = 1 if rows_per_slice == 0 else rows_per_slice

    slices = []
    column_index = 0

    for step in range(0, num_rows, rows_per_slice):
        df_slice = df.iloc[step : step + rows_per_slice]
        col_to_drop_i = (column_index + 1) % num_cols
        if col_to_drop_i != 0:
            col_to_drop = cols[col_to_drop_i]
            df_slice = df_slice.drop(columns=[col_to_drop])
        column_index += 1
        slices.append(df_slice)

    expected = pd.concat(slices)
    return expected, slices


def regularize_dataframe(df):
    output = df.copy(deep=True)
    for col in output.select_dtypes(include=["object"]).columns:
        output[col] = output[col].fillna("")

    # TODO remove this when filtering code returns NaN
    output.fillna(0, inplace=True)
    output = output.reindex(sorted(output.columns), axis=1)
    output = output.reset_index(drop=True)
    output = output.astype("float", errors="ignore")
    return output


@attr.s(slots=True, auto_attribs=True)
class BeforeAfterTimestamp:
    before: pd.Timestamp
    after: Optional[pd.Timestamp]


@contextmanager
def distinct_timestamps(lib: NativeVersionStore):
    """Ensures the timestamp used by ArcticDB operations before, during and leaving the context are all different.

    Yields
    ------
    BeforeAfterTimestamp
    """
    get_ts = lib.version_store.get_store_current_timestamp_for_tests
    before = get_ts()
    while get_ts() == before:
        time.sleep(0.000001)  # 1us - The worst resolution in our clock implementations
    out = BeforeAfterTimestamp(pd.Timestamp(before, unit="ns"), None)
    try:
        yield out
    finally:
        right_after = get_ts()
        while get_ts() == right_after:
            time.sleep(0.000001)
        out.after = pd.Timestamp(get_ts(), unit="ns")


@contextmanager
def random_seed_context():
    seed = os.getenv("ARCTICDB_RAND_SEED")
    state = random.getstate()
    if seed is not None:
        random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


DYNAMIC_STRINGS_SUFFIX = "dynamic_strings"
FIXED_STRINGS_SUFFIX = "fixed_strings"


def generic_filter_test(lib, symbol, arctic_query, expected):
    received = lib.read(symbol, query_builder=arctic_query).data
    if not np.array_equal(expected, received):
        original_df = lib.read(symbol).data
        print(
            f"""Original df:\n{original_df}\nwith dtypes:\n{original_df.dtypes}\nquery:\n{arctic_query}"""
            f"""\nPandas result:\n{expected}\nArcticDB result:\n{received}"""
        )
        assert False


# For string queries, test both with and without dynamic strings, and with the query both optimised for speed and memory
def generic_filter_test_strings(lib, base_symbol, arctic_query, expected):
    for symbol in [f"{base_symbol}_{DYNAMIC_STRINGS_SUFFIX}", f"{base_symbol}_{FIXED_STRINGS_SUFFIX}"]:
        arctic_query.optimise_for_speed()
        generic_filter_test(lib, symbol, arctic_query, expected)
        arctic_query.optimise_for_memory()
        generic_filter_test(lib, symbol, arctic_query, expected)


def generic_filter_test_dynamic(lib, symbol, arctic_query, queried_slices):
    received = lib.read(symbol, query_builder=arctic_query).data
    assert len(received) == sum([len(queried_slice) for queried_slice in queried_slices])
    start_row = 0
    arrays_equal = True
    for queried_slice in queried_slices:
        for col_name in queried_slice.columns:
            if not np.array_equal(
                queried_slice[col_name], received[col_name].iloc[start_row : start_row + len(queried_slice)]
            ):
                arrays_equal = False
        start_row += len(queried_slice)
    if not arrays_equal:
        original_df = lib.read(symbol).data
        print(
            f"""Original df (in ArcticDB, backfilled):\n{original_df}\nwith dtypes:\n{original_df.dtypes}\nquery:\n{arctic_query}"""
            f"""\nPandas result:\n{queried_slices}\nArcticDB result:\n{received}"""
        )
        assert False


# For string queries, test both with and without dynamic strings, and with the query both optimised for speed and memory
def generic_filter_test_strings_dynamic(lib, base_symbol, slices, arctic_query, pandas_query):
    queried_slices = []
    for slice in slices:
        try:
            queried_slices.append(slice.query(pandas_query))
        except UndefinedVariableError:
            # Might have edited out the query columns entirely
            pass
    for symbol in [f"{base_symbol}_{DYNAMIC_STRINGS_SUFFIX}", f"{base_symbol}_{FIXED_STRINGS_SUFFIX}"]:
        arctic_query.optimise_for_speed()
        generic_filter_test_dynamic(lib, symbol, arctic_query, queried_slices)
        arctic_query.optimise_for_memory()
        generic_filter_test_dynamic(lib, symbol, arctic_query, queried_slices)


# TODO: Replace with np.array_equal with equal_nan argument (added in 1.19.0)
def generic_filter_test_nans(lib, symbol, arctic_query, expected):
    received = lib.read(symbol, query_builder=arctic_query).data
    assert expected.shape == received.shape
    for col in expected.columns:
        expected_col = expected.loc[:, col]
        received_col = received.loc[:, col]
        for idx, expected_val in expected_col.items():
            received_val = received_col[idx]
            if isinstance(expected_val, str):
                assert isinstance(received_val, str) and expected_val == received_val
            elif expected_val is None:
                assert received_val is None
            elif np.isnan(expected_val):
                assert np.isnan(received_val)


def generic_aggregation_test(lib, symbol, df, grouping_column, aggs_dict):
    expected = df.groupby(grouping_column).agg(aggs_dict)
    expected = expected.reindex(columns=sorted(expected.columns))
    q = QueryBuilder().groupby(grouping_column).agg(aggs_dict)
    received = lib.read(symbol, query_builder=q).data
    received = received.reindex(columns=sorted(received.columns))
    received.sort_index(inplace=True)
    assert_frame_equal(expected, received, check_dtype=False)


def generic_named_aggregation_test(lib, symbol, df, grouping_column, aggs_dict, agg_dtypes=None):
    expected = df.groupby(grouping_column).agg(None, **aggs_dict)
    expected = expected.reindex(columns=sorted(expected.columns))
    if agg_dtypes is not None:
        assert expected.index.name == "grouping_column"
        expected.index = expected.index.astype(agg_dtypes["grouping_column"])
        del agg_dtypes["grouping_column"]
        for name, dtype in agg_dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                expected[name] = expected[name].fillna(0)
        expected = expected.astype(agg_dtypes)
    q = QueryBuilder().groupby(grouping_column).agg(aggs_dict)
    received = lib.read(symbol, query_builder=q).data
    received = received.reindex(columns=sorted(received.columns))
    received.sort_index(inplace=True)
    try:
        assert_frame_equal(expected, received, check_dtype=agg_dtypes is not None)
    except AssertionError as e:
        print(
            f"""Original df:\n{df}\nwith dtypes:\n{df.dtypes}\naggs dict:\n{aggs_dict}"""
            f"""\nPandas result:\n{expected}\n"ArcticDB result:\n{received}"""
            f"""\n{df.dtypes}"""
            f"""\n{expected.dtypes}"""
            f"""\n{received.dtypes}"""
            f"""\n{agg_dtypes}"""
        )
        raise e


def drop_inf_and_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

def drop_inf(df):
    return df[~df.isin([np.inf, -np.inf]).any(axis=1)]

def assert_dfs_approximate(left: pd.DataFrame, right: pd.DataFrame, check_dtype=False):
    """
    Checks if integer columns are exactly the same. For float columns checks if they are approximately the same.
    We can't guarantee the same order of operations for the floats thus numerical errors might appear.
    """
    message = f"{left.shape} != {right.shape}  ----  Left.index=[{left.index}], Right.index=[{right.index}]"
    assert left.shape == right.shape, message
    message = f"left.columns({left.columns}) != right.columns({right.columns})"
    assert left.columns.equals(right.columns), message
    # To avoid checking the freq member of the index as arctic does not fill it in
    message = f"left.index({left.index}) != right.index({right.index})"
    assert left.index.equals(right.index), message

    # Drop NaN an inf values because. Pandas uses Kahan summation algorithm to improve numerical stability.
    # Thus they don't consistently overflow to infinity. Discussion: https://github.com/pandas-dev/pandas/issues/60303
    left_no_inf = drop_inf(left)
    right_no_inf = drop_inf(right)

    check_equals_flags = {"check_dtype": check_dtype}
    if PANDAS_VERSION >= Version("1.1"):
        check_equals_flags["check_freq"] = False
    if PANDAS_VERSION >= Version("1.2"):
        check_equals_flags["check_flags"] = False
    for col in left_no_inf.columns:
        if pd.api.types.is_integer_dtype(left_no_inf[col].dtype) and pd.api.types.is_integer_dtype(right_no_inf[col].dtype):
            pd.testing.assert_series_equal(left_no_inf[col], right_no_inf[col], **check_equals_flags)
        else:
            if PANDAS_VERSION >= Version("1.1"):
                check_equals_flags["rtol"] = 3e-4
            pd.testing.assert_series_equal(left_no_inf[col], right_no_inf[col], **check_equals_flags)


def generic_resample_test(
        lib,
        sym,
        rule,
        aggregations,
        data,
        date_range=None,
        closed=None,
        label=None,
        offset=None,
        origin=None,
        drop_empty_buckets_for=None,
        expected_types=None,
):
    """
    Perform a resampling in ArcticDB and compare it against the same query in Pandas.

    :param drop_empty_buckets_for: Will add additional aggregation column using the count aggregator. At the end of the
    aggregation query will remove all rows for which this newly added count aggregation is 0. Works only for int/uint
    columns. There is a similar function generic_resample_test_with_empty_buckets in
    python/tests/unit/arcticdb/version_store/test_resample.py which can drop empty buckets for all types of columns,
    but it cannot take parameters such as origin and offset.
    """
    # Pandas doesn't have a good date_range equivalent in resample, so just use read for that
    original_data = data if date_range is None else data.loc[date_range[0]:date_range[-1]]
    # Pandas 1.X needs None as the first argument to agg with named aggregators

    pandas_aggregations = (
        {**aggregations, "_bucket_size_": (drop_empty_buckets_for, "count")} if drop_empty_buckets_for else aggregations
    )
    resample_args = {}
    if origin:
        resample_args["origin"] = origin
    if offset:
        resample_args["offset"] = offset

    if PANDAS_VERSION >= Version("1.1.0"):
        resampler = original_data.resample(rule, closed=closed, label=label, **resample_args)
        try:
            expected = resampler.agg(None, **pandas_aggregations)
        except ValueError:
            bins = resampler.groups.keys()
            if len(bins) == 0:
                # This is due to a bug in Pandas https://github.com/pandas-dev/pandas/issues/44957
                # When none of the values fall in any bucket Pandas, the groups in the resampler are empty, and Pandas
                # throws ValueError. ArcticDB behaves reasonably and returns an empty DataFrame.
                # This seems possible only if the origin is end_day
                if expected_types is not None:
                    _expected_types = copy.deepcopy(expected_types)
                    _expected_types["_bucket_size_"] = np.uint64
                else:
                    _expected_types = None
                expected = pd.DataFrame(
                    {col_name: np.array([], dtype=_expected_types[col_name] if _expected_types else None) for col_name in pandas_aggregations},
                    index=pd.DatetimeIndex([])
                )
            else:
                raise
    else:
        expected = original_data.resample(rule, closed=closed, label=label).agg(None, **pandas_aggregations)
    if drop_empty_buckets_for:
        expected = expected[expected["_bucket_size_"] > 0]
        expected.drop(columns=["_bucket_size_"], inplace=True)
    expected = expected.reindex(columns=sorted(expected.columns))

    if expected_types:
        for name, dtype in expected_types.items():
            if pd.api.types.is_integer_dtype(dtype):
                expected[name] = expected[name].fillna(0)
        expected = expected.astype(expected_types)

    q = QueryBuilder()
    if origin:
        q = q.resample(rule, closed=closed, label=label, offset=offset, origin=origin).agg(aggregations)
    else:
        q = q.resample(rule, closed=closed, label=label, offset=offset).agg(aggregations)
    received = lib.read(sym, date_range=date_range, query_builder=q).data
    received = received.reindex(columns=sorted(received.columns))

    has_float_column = any(pd.api.types.is_float_dtype(col_type) for col_type in list(expected.dtypes))
    check_dtype = expected_types is not None
    if has_float_column:
        assert_dfs_approximate(expected, received, check_dtype=check_dtype)
    else:
        assert_frame_equal(expected, received, check_dtype=check_dtype)


def equals(x, y):
    if isinstance(x, tuple) or isinstance(x, list):
        assert len(x) == len(y)
        for vx, vy in zip(x, y):
            equals(vx, vy)
    elif isinstance(x, dict):
        assert isinstance(y, dict)
        assert set(x.keys()) == set(y.keys())
        for k in x.keys():
            equals(x[k], y[k])
    elif isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)
        assert np.allclose(x, y)
    else:
        assert x == y


def is_pytest_running():
    """Check if code is currently running as part of a pytest test."""
    return "PYTEST_CURRENT_TEST" in os.environ


def common_sum_aggregation_dtype(left, right):
    if pd.api.types.is_signed_integer_dtype(left) and pd.api.types.is_signed_integer_dtype(right):
        return np.int64
    elif pd.api.types.is_unsigned_integer_dtype(left) and pd.api.types.is_unsigned_integer_dtype(right):
        return np.uint64
    elif ((pd.api.types.is_signed_integer_dtype(left) and pd.api.types.is_unsigned_integer_dtype(right)) or
          (pd.api.types.is_unsigned_integer_dtype(left) and pd.api.types.is_signed_integer_dtype(right))):
        return np.int64
    elif pd.api.types.is_bool_dtype(left) and pd.api.types.is_bool_dtype(right):
        return np.uint64
    else:
        return np.float64

def largest_numeric_type(dtype):
    """
    Given a dtype return a dtype of the same category (signed int, unsigned int, float) with the maximum supported by
    ArcticDB byte size.
    """
    if pd.api.types.is_float_dtype(dtype):
        return np.float64
    elif pd.api.types.is_signed_integer_dtype(dtype):
        return np.int64
    elif pd.api.types.is_unsigned_integer_dtype(dtype):
        return np.uint64
    return dtype

def is_numeric_type(dtype):
    return pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)

def valid_common_type(left, right):
    """
    This is created to mimic the C++ has_valid_common_type function. It takes two numpy dtypes and returns a type able
    to represent both or None otherwise.

    This works only with numeric types (int, uint, float)
    """
    if left is None or right is None:
        return None
    left = np.dtype(left)
    right = np.dtype(right)
    if left == right:
        return left

    if pd.api.types.is_bool_dtype(left):
        if pd.api.types.is_bool_dtype(right):
            return left
        elif pd.api.types.is_integer_dtype(right) or pd.api.types.is_float_dtype(right):
            return None
        raise Exception(f"Unexpected right dtype: {right}")
    if pd.api.types.is_float_dtype(left):
        if pd.api.types.is_float_dtype(right) or pd.api.types.is_integer_dtype(right):
            return np.promote_types(left, right)
        elif pd.api.types.is_bool_dtype(right):
            return None
        raise Exception(f"Unexpected right dtype: {right}")
    elif pd.api.types.is_signed_integer_dtype(left):
        if pd.api.types.is_float_dtype(right) or pd.api.types.is_signed_integer_dtype(right):
            return np.promote_types(left, right)
        elif pd.api.types.is_unsigned_integer_dtype(right):
            # Numpy promotes int* and uint64 to float64. ArcticDB does not allow such promotion
            return None if right.itemsize >= 8 else np.promote_types(left, right)
        elif pd.api.types.is_bool_dtype(right):
            return None
        raise Exception(f"Unexpected right dtype: {right}")
    elif pd.api.types.is_unsigned_integer_dtype(left):
        if pd.api.types.is_float_dtype(right) or pd.api.types.is_unsigned_integer_dtype(right):
            return np.promote_types(left, right)
        elif pd.api.types.is_signed_integer_dtype(right):
            # Numpy promotes int* and uint64 to float64. ArcticDB does not allow such promotion
            return None if left.itemsize >= 8 else np.promote_types(left, right)
        elif pd.api.types.is_bool_dtype(right):
            return None
        raise Exception(f"Unexpected right dtype: {right}")
    elif pd.api.types.is_datetime64_any_dtype(left):
        if pd.api.types.is_datetime64_any_dtype(right):
            return np.promote_types(left, right)
        elif is_numeric_type(right) or pd.api.types.is_bool_dtype(right):
            return None
        raise Exception(f"Unexpected right dtype: {right}")
    raise Exception(f"Unexpected left dtype: {left}")

def expected_aggregation_type(aggregation, df_list, column_name):
    common_types = compute_common_type_for_columns_in_df_list(df_list)
    if aggregation == "count":
        return np.uint64
    elif aggregation == "mean":
        return np.float64
    elif aggregation == "sum":
        sum_column_types = [df[column_name].dtype for df in df_list if column_name in df.columns]
        return reduce(common_sum_aggregation_dtype, sum_column_types, sum_column_types[0])
    elif aggregation in ["min", "max", "first", "last"]:
        return common_types[column_name]
    else:
        raise Exception(f"Unknown aggregation type: {aggregation}.")


def compute_common_type_for_columns_in_df_list(df_list):
    common_types = {}
    for df in df_list:
        for col in df.columns:
            if col not in common_types:
                common_types[col] = np.dtype(df[col].dtype)
            else:
                common_types[col] = valid_common_type(common_types[col], np.dtype(df[col].dtype))
    return common_types

def compute_common_type_for_columns(segment_columns: List[dict]):
    """
    Takes a list of column/dtype dictionaries where each element of the list is a dictionary describing a segment. The
    keys of the dictionary are column names and the values are dtypes. A column is allowed to be missing from some
    segments. Returns a dictionary where the keys are column names and values are combined dtype. If a value is none
    this means that there are two segments holding a column with incompatible dtypes.
    """
    common_types = {}
    for columns in segment_columns:
        for name, dtype in columns.items():
            if name not in common_types:
                common_types[name] = np.dtype(dtype)
            else:
                common_types[name] = valid_common_type(common_types[name], np.dtype(dtype))
    return common_types