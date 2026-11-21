"""The run-stage store (kappaccino.io.store): exact round trips, the old-layout fallback, replacement and repacking,
recovery from an unreadable file, concurrent writers and the command line."""
import json
import threading

import h5py
import numpy as np
import pandas as pd
import pytest

from kappaccino.io import store
from kappaccino.io.hdf5 import load_arrays, save_arrays


def _frames():
    rng = np.random.default_rng(7)
    return {
        'ids_and_numbers': pd.DataFrame({'family_id': ['1', '2', '10'], 'image_label': ['1.1', '2.1', '10.3'],
                                         'x': [1.0, np.nan, -0.0], 'n': [1, 2, 3], 'flag': [True, False, True]}),
        'all_dtypes': pd.DataFrame({'i8': np.array([1, -2], np.int8), 'u16': np.array([1, 2], np.uint16),
                                    'i64': np.array([2**62, -1]), 'f32': np.array([.1, np.inf], np.float32),
                                    'f64': [np.pi, -np.inf], 'c': np.array([1 + 2j, 0j]), 'b': [False, True]}),
        'object_mixed': pd.DataFrame({'text_or_none': pd.Series(['1', None, 'x'], dtype=object),
                                      'bool_or_missing': pd.Series([True, None, np.nan], dtype=object),
                                      'mixed': pd.Series([1, 'a', 2.5], dtype=object),
                                      'numpy_scalars': pd.Series([np.int64(3), np.float64(.5), np.bool_(True)], dtype=object),
                                      'all_text': pd.Series(['a', 'b', ''], dtype=object)}),
        'pandas_strings': pd.DataFrame({'str': pd.Series(['a', None, ''], dtype='str'),
                                        'string': pd.Series(['a', None, 'c'], dtype='string')}),
        'nullable': pd.DataFrame({'Int64': pd.array([1, None, 3], dtype='Int64'), 'boolean': pd.array([True, None, False], dtype='boolean'),
                                  'Float64': pd.array([.5, None, np.nan], dtype='Float64')}),
        'datetime': pd.DataFrame({'t': pd.to_datetime(['2020-01-01', None]), 'dt': pd.to_timedelta(['1s', None])}),
        'empty_with_columns': pd.DataFrame(columns=['a', 'b']),
        'empty_typed': pd.DataFrame({'a': pd.Series([], dtype=float), 'b': pd.Series([], dtype=str), 'c': pd.Series([], dtype=bool)}),
        'no_columns': pd.DataFrame(),
        'rows_no_columns': pd.DataFrame(index=range(3)),
        'duplicate_columns': pd.DataFrame([[1, 2.0]], columns=['a', 'a']),
        'odd_column_names': pd.DataFrame({'a/b': [1], 'with space': [2], 'p_ü': [3], '': [4]}),
        'numeric_column_names': pd.DataFrame({0: [1.0], 1: ['x']}),
        'large_random': pd.DataFrame({'x': rng.normal(size=5000), 'k': rng.integers(0, 9, 5000)}),
        'unicode_and_nul': pd.DataFrame({'s': pd.Series(['é', 'a\x00b', '∞'], dtype=object)}),
        'list_and_dict_cells': pd.DataFrame({'kinematics': pd.Series([['sigma'], [], ['sigma', 'r']], dtype=object),
                                             'nested': pd.Series([{'a': [1, 2.5, None]}, None, {'b': 'x'}], dtype=object)}),
    }


@pytest.mark.parametrize('name', list(_frames()))
def test_table_round_trip_is_exact(tmp_path, name):
    frame = _frames()[name]
    store.write_table(tmp_path, name, frame)
    back = store.read_table(tmp_path, name)
    pd.testing.assert_frame_equal(back, frame.reset_index(drop=True) if frame.shape[1] else frame, check_exact=True)
    assert list(back.columns) == list(frame.columns)
    assert list(back.dtypes) == list(frame.dtypes)
    for position in range(frame.shape[1]):
        written, read = frame.iloc[:, position].tolist(), back.iloc[:, position].tolist()
        # Same cell types (str stays str, None stays None, NaN stays a float NaN) and the same float bits.
        assert [type(v) for v in read] == [type(v.item()) if isinstance(v, np.generic) else type(v) for v in written]
        if frame.iloc[:, position].dtype.kind == 'f':
            assert (frame.iloc[:, position].to_numpy().view(np.uint64 if frame.iloc[:, position].dtype == np.float64 else np.uint32)
                    == back.iloc[:, position].to_numpy().view(np.uint64 if frame.iloc[:, position].dtype == np.float64 else np.uint32)).all()


def test_table_index_is_not_stored(tmp_path):
    frame = pd.DataFrame({'a': [3, 1, 2]}).sort_values('a')
    store.write_table(tmp_path, 't', frame)
    pd.testing.assert_frame_equal(store.read_table(tmp_path, 't'), frame.reset_index(drop=True))


@pytest.mark.parametrize('cell', [(1, 2), [1, (2,)], {1: 'a'}, pd.NA, object()])
def test_table_refuses_cells_it_cannot_store_exactly(tmp_path, cell):
    frame = pd.DataFrame({'a': pd.Series(['x', cell], dtype=object)})
    with pytest.raises(TypeError):
        store.write_table(tmp_path, 't', frame)


def test_failed_write_keeps_the_stored_entry(tmp_path):
    good = pd.DataFrame({'a': [1, 2]})
    store.write_table(tmp_path, 't', good)
    with pytest.raises(TypeError):
        store.write_table(tmp_path, 't', pd.DataFrame({'a': pd.Series([(1,)], dtype=object)}))
    pd.testing.assert_frame_equal(store.read_table(tmp_path, 't'), good)
    with h5py.File(tmp_path / store.RESULTS_FILE) as file:
        assert list(file['tables']) == ['t']


def test_record_round_trip(tmp_path):
    record = {'a': 1, 'b': [1.5, None, float('nan'), float('inf')], 'nested': {'x': 'y', 'z': [True, False]}, 'text': 'é∞',
              'pi': np.pi}
    store.write_record(tmp_path, 'r', record)
    back = store.read_record(tmp_path, 'r')
    assert json.dumps(back, allow_nan=True) == json.dumps(record, allow_nan=True)
    assert np.isnan(back['b'][2]) and back['pi'] == np.pi
    for value in ([1, 2], 'text', 3.25, None, []):
        store.write_record(tmp_path, 'scalar', value)
        assert store.read_record(tmp_path, 'scalar') == value


def test_record_conversions_match_the_old_json_files(tmp_path):
    """NumPy values, tuples, paths and non-string keys are written as the old json writers wrote them."""
    record = {'i': np.int64(3), 'f': np.float32(.5), 'b': np.bool_(True), 'arr': np.arange(3), 't': (1, 2), 1: 'one',
              'p': tmp_path}
    store.write_record(tmp_path, 'r', record)
    assert store.read_record(tmp_path, 'r') == {'i': 3, 'f': .5, 'b': True, 'arr': [0, 1, 2], 't': [1, 2], '1': 'one',
                                                 'p': str(tmp_path)}
    # default=float, as the run summary and audit were written: integers become floats, exactly as in the old files.
    summary = {'count': np.int64(5), 'value': np.float64(.1)}
    store.write_record(tmp_path, 'summary', summary, default=float)
    assert store.read_record(tmp_path, 'summary') == json.loads(json.dumps(summary, indent=2, default=float))


def test_arrays_round_trip(tmp_path):
    arrays = dict(x=np.arange(12.).reshape(3, 4), s=np.array(['a', 'bc']), scalar=3.5, flag=np.bool_(True), text='key',
                  skipped=None, empty=np.empty((0, 2)))
    store.write_arrays(tmp_path, 'bundle', arrays)
    back = store.read_arrays(tmp_path, 'bundle')
    assert set(back) == {'x', 's', 'scalar', 'flag', 'text', 'empty'}
    np.testing.assert_array_equal(back['x'], arrays['x'])
    assert back['s'].dtype.kind == 'U' and back['s'].tolist() == ['a', 'bc']
    assert back['scalar'].shape == () and float(back['scalar']) == 3.5 and str(back['text']) == 'key'
    assert back['empty'].shape == (0, 2)
    # The same arrays as save_arrays / load_arrays give.
    save_arrays(tmp_path / 'old.h5', **arrays)
    old = load_arrays(tmp_path / 'old.h5')
    assert set(old) == set(back)
    for key in old:
        assert old[key].dtype == back[key].dtype
        np.testing.assert_array_equal(old[key], back[key])
    with pytest.raises(TypeError):
        store.write_arrays(tmp_path, 'bad', o=np.array([object()]))


def test_figure_records_by_figure_path(tmp_path):
    figure = tmp_path / 'final' / 'figures' / 'hera_corner.pdf'
    store.write_figure_record(figure, {'draws': 3})
    assert store.read_figure_record(figure) == {'draws': 3}
    assert store.has_figure_record(figure)
    with h5py.File(tmp_path / 'final' / store.RESULTS_FILE) as file:
        assert list(file['figure_records']) == ['hera_corner']
    # A figure record and a table record of the same name do not collide.
    store.write_record(tmp_path / 'final', 'hera_corner', {'other': True})
    assert store.read_figure_record(figure) == {'draws': 3}
    assert store.read_record(tmp_path / 'final', 'hera_corner') == {'other': True}


def test_stage_folder_or_its_tables_folder(tmp_path):
    stage = tmp_path / 'final'
    store.write_table(stage / 'tables', 't', pd.DataFrame({'a': [1]}))
    assert (stage / store.RESULTS_FILE).is_file() and not (stage / 'tables').exists()
    assert store.read_table(stage, 't')['a'].tolist() == [1]
    assert store.read_table(stage / 'figures', 't')['a'].tolist() == [1]


def test_missing_entries(tmp_path):
    with pytest.raises(store.MissingEntry):
        store.read_table(tmp_path, 'nothing')
    with pytest.raises(FileNotFoundError):            # callers that caught a missing file still catch it
        store.read_record(tmp_path, 'nothing')
    with pytest.raises(store.MissingEntry):
        store.read_arrays(tmp_path, 'nothing')
    with pytest.raises(store.MissingEntry):
        store.read_cache_arrays(tmp_path, 'nothing')
    with pytest.raises(store.MissingEntry):
        store.read_figure_record(tmp_path / 'figures' / 'x.pdf')
    assert not store.has_table(tmp_path, 'nothing') and not store.has_record(tmp_path, 'nothing')
    with pytest.raises(ValueError):
        store.write_table(tmp_path, 'a/b', pd.DataFrame())


def test_loose_files_are_not_read(tmp_path):
    # results.h5 is the only source: a file under tables/ or figures/ is no entry.
    stage = tmp_path / 'final'
    (stage / 'tables').mkdir(parents=True)
    (stage / 'figures').mkdir()
    (stage / 'tables' / 'image_fit_quality.csv').write_text('family_id,x\n1,0.5\n')
    (stage / 'tables' / 'run_summary.json').write_text('{}')
    (stage / 'figures' / 'hera_corner.json').write_text('{}')
    with pytest.raises(store.MissingEntry):
        store.read_table(stage, 'image_fit_quality')
    with pytest.raises(store.MissingEntry):
        store.read_record(stage, 'run_summary')
    with pytest.raises(store.MissingEntry):
        store.read_figure_record(stage / 'figures' / 'hera_corner.pdf')
    assert not store.has_table(stage, 'image_fit_quality') and not any(store.entries(stage).values())


def test_writing_a_name_again_replaces_it(tmp_path):
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    store.write_table(tmp_path, 't', pd.DataFrame({'b': ['x', 'y']}))
    pd.testing.assert_frame_equal(store.read_table(tmp_path, 't'), pd.DataFrame({'b': ['x', 'y']}))
    store.write_arrays(tmp_path, 'a', x=np.ones(3))
    store.write_arrays(tmp_path, 'a', y=np.zeros(2))
    assert set(store.read_arrays(tmp_path, 'a')) == {'y'}
    with h5py.File(tmp_path / store.RESULTS_FILE) as file:
        assert list(file['tables']) == ['t'] and list(file['arrays']) == ['a']


def test_replaced_space_is_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'REPACK_MIN_BYTES', 1 << 20)
    big = np.random.default_rng(0).normal(size=(400_000,))          # 3.2 MB, incompressible
    for _ in range(6):
        store.write_arrays(tmp_path, 'big', x=big)
    size = (tmp_path / store.RESULTS_FILE).stat().st_size
    assert size < 2.5 * big.nbytes, size                            # repacked on the way, not six copies
    store.write_table(tmp_path, 'small', pd.DataFrame({'a': [1]}))
    store.write_table(tmp_path, 'small', pd.DataFrame({'a': [2]}))
    store.compact(tmp_path)
    with h5py.File(tmp_path / store.RESULTS_FILE) as file:
        assert int(file.attrs.get(store.FREED_ATTRIBUTE, 0)) == 0
    np.testing.assert_array_equal(store.read_arrays(tmp_path, 'big')['x'], big)
    assert store.read_table(tmp_path, 'small')['a'].tolist() == [2]
    assert (tmp_path / store.RESULTS_FILE).stat().st_size < 1.3 * big.nbytes


@pytest.mark.parametrize('cache', [False, True])
def test_unreadable_file_is_moved_aside_on_the_next_write(tmp_path, cache):
    write = store.write_cache_arrays if cache else store.write_arrays
    read = store.read_cache_arrays if cache else store.read_arrays
    path = tmp_path / (store.CACHE_FILE if cache else store.RESULTS_FILE)
    write(tmp_path, 'kept', x=np.ones(2))
    data = bytearray(path.read_bytes())
    data[:8] = b'garbage!'                                          # the superblock signature: unreadable
    path.write_bytes(bytes(data))
    messages = []
    with store.logging_to(messages.append):
        # A reader never moves the file: it logs and stops (results: whether the entry is there cannot be told; it is
        # never reported missing) or treats it as a miss (cache).
        with pytest.raises(store.MissingEntry if cache else store.DamagedEntry):
            read(tmp_path, 'kept')
        if not cache:
            with pytest.raises(store.DamagedEntry):
                store.has_arrays(tmp_path, 'kept')
        assert path.exists() and not list(tmp_path.glob('*.unreadable-*'))
        write(tmp_path, 'new', x=np.zeros(1))
    moved = list(tmp_path.glob(path.name + '.unreadable-*'))
    assert len(moved) == 1 and moved[0].read_bytes() == bytes(data)     # moved, never deleted
    assert any('could not be opened' in m and moved[0].name in m for m in messages)
    assert any(('empty cache' if cache else 'empty results file') in m for m in messages)
    assert set(read(tmp_path, 'new')) == {'x'}
    with pytest.raises(store.MissingEntry):
        read(tmp_path, 'kept')


def test_writes_from_many_threads(tmp_path):
    errors = []

    def work(k):
        try:
            for j in range(5):
                store.write_table(tmp_path, f't{k}', pd.DataFrame({'k': [k] * (j + 1)}))
                store.write_record(tmp_path, f'r{k}', {'k': k, 'j': j})
                store.write_cache_arrays(tmp_path, f'a{k}', x=np.full(10, k))
                store.read_table(tmp_path, f't{k}')
        except Exception as error:              # noqa: BLE001 - reported below
            errors.append(error)
    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    for k in range(8):
        assert store.read_table(tmp_path, f't{k}')['k'].tolist() == [k] * 5
        assert store.read_record(tmp_path, f'r{k}') == {'k': k, 'j': 4}
        np.testing.assert_array_equal(store.read_cache_arrays(tmp_path, f'a{k}')['x'], np.full(10, k))


def test_cache_is_separate_from_results(tmp_path):
    store.write_cache_arrays(tmp_path, 'zeta', x=np.ones(2))
    store.write_cache_record(tmp_path, 'meta', {'key': 'k'})
    store.write_cache_table(tmp_path, 'rows', pd.DataFrame({'a': ['1']}))
    assert not (tmp_path / store.RESULTS_FILE).exists()
    assert store.read_cache_record(tmp_path, 'meta') == {'key': 'k'}
    assert store.read_cache_table(tmp_path, 'rows')['a'].tolist() == ['1']
    assert store.has_cache_arrays(tmp_path, 'zeta') and not store.has_arrays(tmp_path, 'zeta')
    (tmp_path / store.CACHE_FILE).unlink()
    assert not store.has_cache_arrays(tmp_path, 'zeta')


def _run(tmp_path):
    run = tmp_path / 'run'
    final, init = run / 'final', run / 'init'
    store.write_table(final, 'image_fit_quality', pd.DataFrame({'family_id': ['1'], 'x': [0.25]}))
    store.write_record(final, 'run_summary', {'fit': {'n': 1}})
    store.write_arrays(final, 'band', determinant=np.eye(2))
    store.write_figure_record(final / 'figures' / 'hera_corner.pdf', {'draws': 2})
    store.write_cache_arrays(final, 'zeta_masses_cache', draws_key='k')
    store.write_record(init, 'shallow_fold_pairs', {'pairs': []})
    return run


def test_command_line_lists_prints_and_exports(tmp_path, capsys):
    run = _run(tmp_path)
    assert store.main([str(run)]) == 0
    listing = capsys.readouterr().out
    for text in ('image_fit_quality', '1 rows x 2 columns', 'run_summary', 'band', 'determinant (2, 2)', 'hera_corner',
                 'zeta_masses_cache', 'shallow_fold_pairs', '[cache]'):
        assert text in listing, text
    assert store.main([str(run), '--table', 'image_fit_quality']) == 0
    assert capsys.readouterr().out == 'family_id,x\n1,0.25\n'
    assert store.main([str(run / 'final'), '--record', 'run_summary']) == 0
    assert json.loads(capsys.readouterr().out) == {'fit': {'n': 1}}
    assert store.main([str(run), '--stage', 'init', '--record', 'shallow_fold_pairs']) == 0
    assert json.loads(capsys.readouterr().out) == {'pairs': []}
    assert store.main([str(run), '--figure-record', 'hera_corner']) == 0
    assert json.loads(capsys.readouterr().out) == {'draws': 2}
    assert store.main([str(run), '--arrays', 'zeta_masses_cache', '--cache']) == 0
    assert 'draws_key' in capsys.readouterr().out
    assert store.main([str(run), '--table', 'missing']) == 1
    out = tmp_path / 'export'
    assert store.main([str(run), '--export', str(out)]) == 0
    assert (out / 'final/tables/image_fit_quality.csv').read_text() == 'family_id,x\n1,0.25\n'
    assert json.loads((out / 'final/tables/run_summary.json').read_text()) == {'fit': {'n': 1}}
    np.testing.assert_array_equal(load_arrays(out / 'final/tables/band.h5')['determinant'], np.eye(2))
    assert json.loads((out / 'final/figures/hera_corner.json').read_text()) == {'draws': 2}
    assert (out / 'final/cache/zeta_masses_cache.h5').is_file()
    assert json.loads((out / 'init/tables/shallow_fold_pairs.json').read_text()) == {'pairs': []}


# ------------------------------------------------------------------------------------------- failure handling

import errno
import fcntl
import os
import random
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

SRC = str(Path(store.__file__).resolve().parents[2])          # the tree this test imports, for the subprocesses


def _python(code, *args, **kwargs):
    env = {**os.environ, 'PYTHONPATH': SRC, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    return subprocess.Popen([sys.executable, '-c', textwrap.dedent(code), *map(str, args)], env=env, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def _wait_for(process, text):
    line = process.stdout.readline()
    assert text in line, (line, process.stderr.read() if process.poll() is not None else '')


def test_a_file_open_in_this_process_is_never_moved_aside(tmp_path):
    """Opening results.h5 read-only elsewhere in the process used to make the next write move it aside."""
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    messages = []
    with h5py.File(tmp_path / store.RESULTS_FILE, 'r') as held, store.logging_to(messages.append):
        store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))
        assert list(held['tables']) == ['kept']                     # the holder keeps its consistent old version
        assert store.read_table(tmp_path, 'kept')['a'].tolist() == [1]
        assert store.has_table(tmp_path, 'new')
    assert store.read_table(tmp_path, 'new')['b'].tolist() == [2]
    assert not list(tmp_path.glob('*.unreadable-*')) and not messages


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores file permissions')
def test_a_read_only_results_file_raises_and_is_left_alone(tmp_path):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()
    path.chmod(0o444)
    try:
        with pytest.raises(store.StoreError, match='read-only'):
            store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))
        assert path.read_bytes() == before and not list(tmp_path.glob('*.unreadable-*'))
        assert store.read_table(tmp_path, 'kept')['a'].tolist() == [1]
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize('code', [errno.EMFILE, errno.EIO])
def test_other_open_errors_raise_and_leave_the_file(tmp_path, monkeypatch, code):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()

    def failing(path):
        raise OSError(code, os.strerror(code))
    monkeypatch.setattr(store, '_open_for_reading', failing)
    with pytest.raises(store.StoreError) as raised:
        store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))
    assert os.strerror(code) in str(raised.value) and 'left as it is' in str(raised.value)
    with pytest.raises(store.StoreError):                           # a reader is told too, never "missing"
        store.read_table(tmp_path, 'kept')
    monkeypatch.undo()
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == [store.RESULTS_FILE,
                                                                                        store.RESULTS_FILE + '.lock']


def test_a_full_disk_while_copying_leaves_the_file(tmp_path, monkeypatch):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()

    def full(source, target):
        Path(target).write_bytes(b'partial')
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
    monkeypatch.setattr(store.shutil, 'copyfile', full)
    with pytest.raises(store.StoreError, match='No space left'):
        store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))
    monkeypatch.undo()
    assert path.read_bytes() == before and not list(tmp_path.glob('*.tmp-*'))


def test_a_truncated_results_file_is_moved_aside_loudly(tmp_path):
    store.write_arrays(tmp_path, 'kept', x=np.random.default_rng(1).normal(size=50_000))
    path = tmp_path / store.RESULTS_FILE
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    messages = []
    with store.logging_to(messages.append):
        with pytest.raises(store.DamagedEntry):
            store.read_arrays(tmp_path, 'kept')
        assert path.exists() and not list(tmp_path.glob('*.unreadable-*'))      # readers never move anything
        store.write_arrays(tmp_path, 'new', x=np.zeros(1))
    moved = list(tmp_path.glob(path.name + '.unreadable-*'))
    assert len(moved) == 1 and moved[0].read_bytes() == data[: len(data) // 2]
    assert any('truncated' in m and 'LOST' in m and moved[0].name in m for m in messages), messages


# Batches -----------------------------------------------------------------------------------------------------

def test_a_batch_is_one_commit_and_sees_its_own_writes(tmp_path, monkeypatch):
    replaced = []
    real = os.replace
    monkeypatch.setattr(store.os, 'replace', lambda a, b: (replaced.append(Path(b).name), real(a, b)))
    with store.batch(tmp_path):
        assert not (tmp_path / store.RESULTS_FILE).exists()         # nothing until the first write
        for k in range(5):
            store.write_table(tmp_path, f't{k}', pd.DataFrame({'k': [k]}))
            store.write_figure_record(tmp_path / 'figures' / f'f{k}.pdf', {'k': k})
            assert store.read_table(tmp_path, f't{k}')['k'].tolist() == [k]
            assert store.has_figure_record(tmp_path / 'figures' / f'f{k}.pdf')
        with store.batch(tmp_path / 'tables'):                      # nested: joins the outer batch
            store.write_record(tmp_path, 'r', {'n': 1})
        assert set(store.entries(tmp_path)[store.TABLES]) == {f't{k}' for k in range(5)}
        assert not (tmp_path / store.RESULTS_FILE).exists() and replaced == []
    assert replaced == [store.RESULTS_FILE]
    assert store.read_record(tmp_path, 'r') == {'n': 1} and len(store.entries(tmp_path)[store.FIGURE_RECORDS]) == 5
    with store.batch(tmp_path):
        pass                                                        # a batch without writes commits nothing
    assert replaced == [store.RESULTS_FILE]


def test_an_exception_in_a_batch_leaves_the_file_as_it_was(tmp_path):
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        with store.batch(tmp_path):
            store.write_table(tmp_path, 't', pd.DataFrame({'a': [2]}))
            store.write_table(tmp_path, 'u', pd.DataFrame({'a': [3]}))
            raise RuntimeError('stage failed')
    assert path.read_bytes() == before and not list(tmp_path.glob('*.tmp-*'))
    # A refused entry inside a batch does not spoil the batch.
    with store.batch(tmp_path):
        with pytest.raises(TypeError):
            store.write_table(tmp_path, 't', pd.DataFrame({'a': pd.Series([(1,)], dtype=object)}))
        store.write_table(tmp_path, 'u', pd.DataFrame({'a': [3]}))
    assert store.read_table(tmp_path, 't')['a'].tolist() == [1] and store.read_table(tmp_path, 'u')['a'].tolist() == [3]
    assert sorted(store.entries(tmp_path)[store.TABLES]) == ['t', 'u']


# Crashes -----------------------------------------------------------------------------------------------------

_CRASH_SERVER = '''
    import contextlib, json, os, random, sys
    import numpy as np, pandas as pd
    from kappaccino.io import store
    batch = getattr(store, 'batch', lambda stage: contextlib.nullcontext())   # (so that the test also runs on old code)
    store.REPACK_MIN_BYTES = 1                    # rebuild (repack) often, so kills land in rebuilds too
    store.set_log(lambda message: None)
    stage = sys.argv[1]
    NAMES = [f'e{i}' for i in range(12)]

    def write(kind, name, version):
        rows = 1 + version % 7
        if kind == 'tables':
            store.write_table(stage, name, pd.DataFrame({'v': [version] * rows, 's': [f'{name}-{version}'] * rows}))
        elif kind == 'records':
            store.write_record(stage, name, {'name': name, 'version': version, 'pad': 'x' * (version % 50)})
        elif kind == 'arrays':
            store.write_arrays(stage, name, v=np.full(rows * 100, version))
        else:
            store.write_figure_record(os.path.join(stage, 'figures', name + '.pdf'), {'version': version})

    def child(out, seed):
        rng = random.Random(seed)
        version = seed * 100000
        while True:
            version += 1
            ops = [(rng.choice(store.KINDS), rng.choice(NAMES), version + k) for k in range(rng.choice([1, 1, 2, 4]))]
            version += len(ops)
            rebuild = rng.random() < 0.1
            os.write(out, ('B ' + json.dumps([] if rebuild else ops) + '\\n').encode())
            if rebuild:
                store.compact(stage)
            elif len(ops) == 1:
                write(*ops[0])
            else:
                with batch(stage):
                    for op in ops:
                        write(*op)
            os.write(out, b'C\\n')

    for line in sys.stdin:
        seed = int(line)
        read, out = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read)
            try:
                child(out, seed)
            finally:
                os._exit(1)
        os.close(out)
        print(pid, flush=True)
        with os.fdopen(read) as messages:
            for message in messages:
                sys.stdout.write(message)
        os.waitpid(pid, 0)
        print('END', flush=True)
'''


def _version(kind, value):
    if kind == store.TABLES:
        return int(value['v'].iloc[0]) if (value['v'] == value['v'].iloc[0]).all() and (value['s'] == value['s'].iloc[0]).all() else None
    if kind == store.ARRAYS:
        return int(value['v'][0]) if (value['v'] == value['v'][0]).all() else None
    return int(value['version'])


def _expected_value_ok(kind, name, version, value):
    rows = 1 + version % 7
    if kind == store.TABLES:
        return value.equals(pd.DataFrame({'v': [version] * rows, 's': [f'{name}-{version}'] * rows}))
    if kind == store.RECORDS:
        return value == {'name': name, 'version': version, 'pad': 'x' * (version % 50)}
    if kind == store.ARRAYS:
        return set(value) == {'v'} and np.array_equal(value['v'], np.full(rows * 100, version))
    return value == {'version': version}


def _read_entry(stage, kind, name):
    if kind == store.FIGURE_RECORDS:
        return store.read_figure_record(stage / 'figures' / f'{name}.pdf')
    return {store.TABLES: store.read_table, store.RECORDS: store.read_record, store.ARRAYS: store.read_arrays}[kind](stage, name)


def test_a_writer_killed_at_random_times_never_damages_committed_entries(tmp_path):
    """SIGKILL a writer (single writes, batches and rebuilds) at random times, 120 times.  After each kill the file
    opens, every committed entry reads back exactly, the batch in flight is all there or not at all, and listing and
    exporting work."""
    trials, rng = 120, random.Random(2026)
    stage = tmp_path / 'final'
    server = _python(_CRASH_SERVER, stage)
    expected, failures, kills_in_flight = {}, [], 0
    try:
        for trial in range(trials):
            server.stdin.write(f'{trial + 1}\n')
            server.stdin.flush()
            pid = int(server.stdout.readline())
            time.sleep(rng.uniform(0.0, 0.08))
            os.kill(pid, signal.SIGKILL)
            in_flight = None
            for line in iter(server.stdout.readline, ''):
                if line.startswith('END'):
                    break
                if line.startswith('B '):
                    in_flight = json.loads(line[2:])
                elif line.startswith('C'):
                    for kind, name, version in in_flight:
                        expected[(kind, name)] = version
                    in_flight = None
            # The file opens, and every entry is the committed version or (all of them or none) the batch in flight.
            try:
                listed = store.entries(stage)
                names = {(kind, name) for kind, found in listed.items() for name in found}
                flight = {(kind, name): version for kind, name, version in (in_flight or [])}
                versions = {key: _version(key[0], _read_entry(stage, *key)) for key in names}
                landed = bool(flight) and all(versions.get(key) == version for key, version in flight.items())
                if landed:
                    expected.update(flight)
                    kills_in_flight += 1
                if names != set(expected):
                    failures.append((trial, 'listed', sorted(names ^ set(expected))))
                for key, version in expected.items():
                    if not _expected_value_ok(key[0], key[1], version, _read_entry(stage, *key)):
                        failures.append((trial, 'value', key, version, versions.get(key)))
                if trial % 10 == 0 and stage.is_dir():         # (no folder yet if the first kill came before any write)
                    store.export(stage, tmp_path / f'export{trial}')
                    assert store.main([str(stage)]) == 0
            except Exception as error:          # noqa: BLE001 - counted below
                failures.append((trial, 'error', repr(error)))
    finally:
        server.stdin.close()
        server.wait(timeout=30)
    assert not failures, failures[:5]
    assert len(expected) > 20                                     # the trials did write and replace entries
    # The next writer removes what the killed ones left behind; nothing else was ever listed as an entry.
    store.write_record(stage, 'last', {'ok': True})
    assert not list(stage.glob('*.tmp-*')) and not list(stage.glob('*.repack')) and not list(stage.glob('*.unreadable-*'))


# Other processes ---------------------------------------------------------------------------------------------

def test_readers_in_other_processes_do_not_block_or_miss_a_commit(tmp_path):
    stage = tmp_path
    store.write_table(stage, 'a', pd.DataFrame({'x': [1.5]}))
    holder = _python('''
        import sys, time, h5py
        with h5py.File(sys.argv[1], 'r') as file:          # HDF5's own file locking, as any h5py user would
            print('open', flush=True)
            sys.stdin.readline()
            print(sorted(file['tables']), flush=True)
    ''', stage / store.RESULTS_FILE)
    poller = _python('''
        import sys, time
        from kappaccino.io import store
        stage, seen, problems = sys.argv[1], 0, []
        print('ready', flush=True)
        end = time.monotonic() + 3
        while time.monotonic() < end:
            try:
                assert store.has_table(stage, 'a')
                assert store.read_table(stage, 'a')['x'].tolist() == [1.5]
                seen += 1
            except Exception as error:
                problems.append(repr(error))
        print(seen, len(problems), problems[:3], flush=True)
    ''', stage)
    try:
        _wait_for(holder, 'open')
        _wait_for(poller, 'ready')
        start = time.monotonic()
        for k in range(30):
            store.write_table(stage, 'b', pd.DataFrame({'k': [k]}))
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, elapsed                                 # no waiting on the holder's HDF5 lock
        holder.stdin.write('\n')
        holder.stdin.flush()
        assert holder.stdout.readline().strip() == "['a']"           # the holder's version stayed consistent
        seen, problems = poller.stdout.readline().split(' ', 1)
        assert int(seen) > 0 and problems.startswith('0 '), (seen, problems, poller.stderr.read())
    finally:
        for process in (holder, poller):
            process.kill()
            process.wait()
    assert store.read_table(stage, 'b')['k'].tolist() == [29]


def test_a_writer_in_another_process_makes_lookups_wait_never_miss(tmp_path, monkeypatch):
    stage = tmp_path
    store.write_table(stage, 'a', pd.DataFrame({'x': ['1']}))
    writer = _python('''
        import sys, time
        import pandas as pd
        from kappaccino.io import store
        with store.batch(sys.argv[1]):
            store.write_table(sys.argv[1], 'w', pd.DataFrame({'y': [2]}))
            print('holding', flush=True)
            sys.stdin.readline()
        print('done', flush=True)
    ''', stage)
    try:
        _wait_for(writer, 'holding')
        start = time.monotonic()
        assert store.has_table(stage, 'a') and not store.has_table(stage, 'w')
        assert store.read_table(stage, 'a')['x'].tolist() == ['1']
        assert set(store.entries(stage)[store.TABLES]) == {'a'}
        assert time.monotonic() - start < 1.0
        # A writer waits for the lock; past the timeout it says the file is busy, and writes nothing.
        monkeypatch.setattr(store, 'WRITER_TIMEOUT', 0.3)
        with pytest.raises(store.StoreBusy, match='busy'):
            store.write_table(stage, 'mine', pd.DataFrame({'z': [3]}))
        assert isinstance(store.StoreBusy(), BlockingIOError)
        monkeypatch.setattr(store, 'WRITER_TIMEOUT', 60.0)
        monkeypatch.setattr(store, 'WAIT_LOG_AFTER', 0.2)
        messages = []
        threading.Timer(0.6, lambda: (writer.stdin.write('\n'), writer.stdin.flush())).start()
        with store.logging_to(messages.append):
            store.write_table(stage, 'mine', pd.DataFrame({'z': [3]}))
        assert any('waiting for' in m and f'pid {writer.pid}' in m for m in messages), messages
        assert writer.stdout.readline().strip() == 'done'
    finally:
        writer.kill()
        writer.wait()
    # Neither write is lost.
    assert set(store.entries(stage)[store.TABLES]) == {'a', 'w', 'mine'}
    assert store.read_table(stage, 'w')['y'].tolist() == [2] and store.read_table(stage, 'mine')['z'].tolist() == [3]


# Temporaries and rebuilding ----------------------------------------------------------------------------------

def test_stale_temporaries_are_removed_only_when_their_writer_is_gone(tmp_path):
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    dead = subprocess.Popen(['true'])
    dead.wait()
    host = store._HOST
    stale = tmp_path / f'{store.RESULTS_FILE}.tmp-{host}-{dead.pid}-0000abcd'
    live = tmp_path / f'{store.RESULTS_FILE}.tmp-{host}-{os.getppid()}-0000abcd'
    remote = tmp_path / f'{store.RESULTS_FILE}.tmp-otherhost-{dead.pid}-0000abcd'
    old_repack, cache_stale = tmp_path / f'{store.RESULTS_FILE}.repack', tmp_path / f'{store.CACHE_FILE}.tmp-{host}-{dead.pid}-1'
    for path in (stale, live, remote, old_repack, cache_stale):
        path.write_bytes(b'partial')
    os.utime(old_repack, (time.time() - 2 * store.STALE_AGE,) * 2)
    messages = []
    with store.logging_to(messages.append):
        store.write_table(tmp_path, 'u', pd.DataFrame({'a': [2]}))
        store.write_cache_record(tmp_path, 'c', {'k': 1})
    assert not stale.exists() and not old_repack.exists() and not cache_stale.exists()
    assert live.exists() and remote.exists()                        # a live process's, another host's (young)
    assert any(stale.name in m for m in messages)
    assert sorted(store.entries(tmp_path)[store.TABLES]) == ['t', 'u']


def test_rebuilding_reclaims_space_without_a_separate_repack_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'REPACK_MIN_BYTES', 1 << 20)
    big = np.random.default_rng(0).normal(size=(400_000,))
    seen = set()
    real = h5py.File

    def spy(name, *args, **kwargs):
        seen.add(Path(name).name.split('.tmp-')[0] + ('.tmp' if '.tmp-' in Path(name).name else ''))
        return real(name, *args, **kwargs)
    monkeypatch.setattr(store.h5py, 'File', spy)
    for _ in range(6):
        store.write_arrays(tmp_path, 'big', x=big)
    monkeypatch.setattr(store.h5py, 'File', real)
    assert (tmp_path / store.RESULTS_FILE).stat().st_size < 2.5 * big.nbytes
    assert sorted(p.name for p in tmp_path.iterdir()) == [store.RESULTS_FILE, store.RESULTS_FILE + '.lock']
    assert 'results.h5.tmp' in seen and 'results.h5.repack' not in seen


# Codec -------------------------------------------------------------------------------------------------------

def test_nullable_float_keeps_nan_apart_from_na(tmp_path):
    values = pd.arrays.FloatingArray(np.array([np.nan, 1.0, 0.0, -0.0]), np.array([False, False, True, False]))
    frame = pd.DataFrame({'f': values, 'f32': pd.arrays.FloatingArray(np.array([np.nan, 2.0], np.float32).repeat(2),
                                                                      np.array([False, True, False, False]))})
    store.write_table(tmp_path, 't', frame)
    back = store.read_table(tmp_path, 't')
    assert list(back.dtypes) == list(frame.dtypes)
    for column in frame:
        assert back[column].array._mask.tolist() == frame[column].array._mask.tolist()
        written = frame[column].array._data[~frame[column].array._mask]
        read = back[column].array._data[~back[column].array._mask]
        assert written.tobytes() == read.tobytes()                  # NaN stays NaN, -0.0 stays -0.0
    assert pd.isna(back['f'][0]) and back['f'].array._mask[0] == False   # noqa: E712 - a real NaN, not NA


def test_record_keys_are_written_as_json_writes_them(tmp_path):
    record = {None: 1, True: 2, False: 3, 1.5: 4, float('nan'): 5, float('inf'): 6, 7: 7, np.int64(8): 8, 'k': 9}
    store.write_record(tmp_path, 'r', record)
    assert store.read_record(tmp_path, 'r') == json.loads(json.dumps({k.item() if isinstance(k, np.generic) else k: v
                                                                      for k, v in record.items()}))
    assert set(store.read_record(tmp_path, 'r')) == {'null', 'true', 'false', '1.5', 'NaN', 'Infinity', '7', '8', 'k'}
    for bad in ({(1, 2): 'tuple key'}, {frozenset(): 1}, {'x': {1, 2}}):
        with pytest.raises(TypeError):
            store.write_record(tmp_path, 'bad', bad)
        with pytest.raises(TypeError):
            store.write_record(tmp_path, 'bad', bad, default=float)
    assert not store.has_record(tmp_path, 'bad')


@pytest.mark.parametrize('dtype', ['str', 'string', object])
def test_strings_with_nul_and_lone_surrogates_round_trip_exactly(tmp_path, dtype):
    cells = ['a\x00b', None, '', 'plain'] + (['\ud800'] if dtype is object else [])   # pyarrow strings refuse surrogates
    frame = pd.DataFrame({'s': pd.Series(cells, dtype=dtype)})
    store.write_table(tmp_path, 't', frame)
    back = store.read_table(tmp_path, 't')
    pd.testing.assert_frame_equal(back, frame, check_exact=True)
    assert back['s'].dtype == frame['s'].dtype


def test_unstorable_input_is_always_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        store.write_arrays(tmp_path, 'a', s=np.array(['a\x00b']))
    with pytest.raises(TypeError):
        store.write_cache_arrays(tmp_path, 'a', s=np.array(['a\x00b']))       # a caller's bug: raised, not cached
    with pytest.raises(TypeError):
        store.write_table(tmp_path, 't', pd.DataFrame({'c': pd.Categorical(['x'])}))
    assert not (tmp_path / store.RESULTS_FILE).exists() or not store.entries(tmp_path)[store.ARRAYS]


def test_tables_keep_their_in_memory_dtypes_not_the_csv_ones(tmp_path):
    """Numeric-looking ids written as strings read back as strings (a CSV would read them as integers)."""
    frame = pd.DataFrame({'family_id': ['1', '2', '10'], 'image_label': ['1.1', '2.1', '10.3']})
    store.write_table(tmp_path, 'image_fit_quality', frame)
    back = store.read_table(tmp_path, 'image_fit_quality')
    assert back['family_id'].tolist() == ['1', '2', '10'] and back['family_id'].dtype == frame['family_id'].dtype
    assert back['family_id'].dtype.kind not in 'iuf'


# Damaged entries ---------------------------------------------------------------------------------------------

def _damage_entries(path, *keys):
    """Overwrite the object headers of ``kind/name`` entries (what a crash in the middle of an in-place write did) and
    leave a ``.writing`` leftover, as files written in place by the first version of the store can hold."""
    with h5py.File(path, 'a') as file:
        file.require_group('tables').create_group('half.writing')
        addresses = [h5py.h5o.get_info(file[key].id).addr for key in keys]
    data = bytearray(path.read_bytes())
    for address in addresses:
        data[address:address + 16] = b'\xff' * 16
    path.write_bytes(bytes(data))


def test_a_damaged_entry_is_reported_by_name_and_the_rest_carries_on(tmp_path, capsys):
    run = _run(tmp_path)
    final = run / 'final'
    store.write_table(final, 'broken_table', pd.DataFrame({'a': [1]}))
    store.write_record(final, 'broken_record', {'b': 2})
    _damage_entries(final / store.RESULTS_FILE, 'tables/broken_table', 'records/broken_record')
    messages = []
    with store.logging_to(messages.append):
        listed = store.entries(final)
        assert 'half.writing' not in listed[store.TABLES]
        assert {'image_fit_quality', 'broken_table'} <= set(listed[store.TABLES])
        with pytest.raises(store.DamagedEntry, match='broken_record'):
            store.read_record(final, 'broken_record')
        with pytest.raises(store.DamagedEntry, match='broken_table') as raised:
            store.read_table(final, 'broken_table')
        # Not "missing": callers that carry on without a missing product stop here instead.
        assert not isinstance(raised.value, (FileNotFoundError, OSError))
        assert store.has_table(final, 'broken_table') and store.has_record(final, 'broken_record')
        assert store.read_record(final, 'run_summary') == {'fit': {'n': 1}}
        written = store.export(run, tmp_path / 'export')
        assert (tmp_path / 'export/final/tables/image_fit_quality.csv') in written
        assert not (tmp_path / 'export/final/tables/broken_table.csv').exists()
        assert any('broken_record' in m and 'not exported' in m for m in messages)
        assert store.main([str(run)]) == 1
        listing = capsys.readouterr().out
        assert 'DAMAGED' in listing and 'broken_table' in listing and '1 rows x 2 columns' in listing
        assert store.main([str(run), '--export', str(tmp_path / 'export2')]) == 1
        assert 'broken_record' in capsys.readouterr().err
        assert store.main([str(run), '--record', 'broken_record']) == 1
        # Writing goes on.  Writing a damaged name again would need a rebuild without it, which would also drop the
        # other damaged entry: refused, and nothing is dropped.
        store.write_table(final, 'after', pd.DataFrame({'c': [3]}))
        assert store.read_table(final, 'after')['c'].tolist() == [3]
        before = (final / store.RESULTS_FILE).read_bytes()
        with pytest.raises(store.StoreError, match='broken_record'):
            store.write_table(final, 'broken_table', pd.DataFrame({'a': [4]}))
        assert (final / store.RESULTS_FILE).read_bytes() == before
    assert not any('LOST' in m for m in messages)
    assert set(store.entries(final)[store.TABLES]) == {'image_fit_quality', 'after', 'broken_table'}
    assert set(store.entries(final)[store.RECORDS]) == {'run_summary', 'broken_record'}
    assert store.read_table(final, 'image_fit_quality')['x'].tolist() == [0.25]
    assert not list(final.glob('*.tmp-*'))


def test_writing_a_damaged_name_drops_only_that_entry_loudly(tmp_path):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    store.write_record(tmp_path, 'record', {'b': 2})
    store.write_table(tmp_path, 'broken', pd.DataFrame({'a': [9]}))
    _damage_entries(tmp_path / store.RESULTS_FILE, 'tables/broken')
    messages = []
    with store.logging_to(messages.append):
        store.write_table(tmp_path, 'broken', pd.DataFrame({'a': [4]}))
    lost = [m for m in messages if 'LOST' in m]
    assert len(lost) == 1 and 'ERROR' in lost[0] and 'tables/broken' in lost[0], messages
    assert store.read_table(tmp_path, 'broken')['a'].tolist() == [4]
    assert store.read_table(tmp_path, 'kept')['a'].tolist() == [1] and store.read_record(tmp_path, 'record') == {'b': 2}
    assert set(store.entries(tmp_path)[store.TABLES]) == {'kept', 'broken'}


def test_a_damaged_slot_whose_stored_entry_reads_is_not_dropped(tmp_path, monkeypatch):
    """The entry cannot be removed from the new version, but a second read of results.h5 succeeds: nothing is
    dropped and nothing is written."""
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [2]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()
    real = store._replace_slot

    def flaky(file, kind, name):
        if name == 't':
            raise store._DamagedSlot(kind, name, KeyError('simulated: cannot unlink'))
        return real(file, kind, name)
    monkeypatch.setattr(store, '_replace_slot', flaky)
    messages = []
    with store.logging_to(messages.append), pytest.raises(store.StoreError, match='second'):
        store.write_table(tmp_path, 't', pd.DataFrame({'a': [3]}))
    assert path.read_bytes() == before and not any('LOST' in m for m in messages)
    assert not list(tmp_path.glob('*.tmp-*'))


# Cache -------------------------------------------------------------------------------------------------------

def test_cache_failures_are_misses_never_errors(tmp_path, monkeypatch):
    messages = []
    with store.logging_to(messages.append):
        (tmp_path / store.CACHE_FILE).mkdir()                        # cannot be opened, and is not "damaged"
        store.write_cache_arrays(tmp_path, 'z', x=np.ones(2))       # logged, not raised
        assert (tmp_path / store.CACHE_FILE).is_dir()               # and not moved aside
        with pytest.raises(store.MissingEntry):
            store.read_cache_arrays(tmp_path, 'z')
        (tmp_path / store.CACHE_FILE).rmdir()
        store.write_cache_arrays(tmp_path, 'z', x=np.ones(2))
        store.write_cache_record(tmp_path, 'meta', {'k': 1})
        _damage_entries(tmp_path / store.CACHE_FILE, 'arrays/z')
        with pytest.raises(store.MissingEntry):
            store.read_cache_arrays(tmp_path, 'z')
        assert store.has_cache_arrays(tmp_path, 'z')                # there, though it cannot be read (a miss)
        assert store.read_cache_record(tmp_path, 'meta') == {'k': 1}
        store.write_cache_arrays(tmp_path, 'z', x=np.zeros(3))      # replaces the damaged entry
        np.testing.assert_array_equal(store.read_cache_arrays(tmp_path, 'z')['x'], np.zeros(3))
        # Another process (here: another open file description) holding the cache lock is a miss, not a wait.
        monkeypatch.setattr(store, 'CACHE_TIMEOUT', 0.2)
        with open(tmp_path / (store.CACHE_FILE + '.lock'), 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with pytest.raises(store.MissingEntry):
                store.read_cache_arrays(tmp_path, 'z')
            store.write_cache_arrays(tmp_path, 'other', x=np.ones(1))
        assert not store.has_cache_arrays(tmp_path, 'other')
    assert any('not written' in m for m in messages) and any('busy' in m for m in messages)


# Rebuilding never loses an entry (review 2, finding 1) -------------------------------------------------------

def _flaky_copy(monkeypatch, victim, error):
    """h5py's object copy fails once, for the entry ``victim`` (a transient read error, or no memory)."""
    real, state = h5py.Group.copy, {'failed': 0}

    def flaky(self, source, dest, name=None, **kwargs):
        if name == victim and not state['failed']:
            state['failed'] += 1
            raise error
        return real(self, source, dest, name=name, **kwargs)
    monkeypatch.setattr(h5py.Group, 'copy', flaky)
    return state


@pytest.mark.parametrize('error', [OSError('Unable to copy object (simulated transient read error)'),
                                   MemoryError('simulated')])
def test_a_failed_copy_gives_up_the_automatic_rebuild_and_keeps_every_entry(tmp_path, monkeypatch, error):
    monkeypatch.setattr(store, 'REPACK_MIN_BYTES', 1 << 20)
    store.write_record(tmp_path, 'precious', {'answer': 42})
    big = np.random.default_rng(0).normal(size=400_000)
    store.write_arrays(tmp_path, 'big', x=big)
    store.write_arrays(tmp_path, 'big', x=big)                      # half the file is now freed space
    state = _flaky_copy(monkeypatch, 'precious', error)
    messages = []
    with store.logging_to(messages.append):
        store.write_record(tmp_path, 'unrelated', {'v': 1})        # would rebuild the file
    monkeypatch.undo()
    assert state['failed'] == 1
    assert store.read_record(tmp_path, 'precious') == {'answer': 42}
    assert store.read_record(tmp_path, 'unrelated') == {'v': 1}
    np.testing.assert_array_equal(store.read_arrays(tmp_path, 'big')['x'], big)
    assert any('not rebuilt' in m and 'precious' in m for m in messages), messages
    assert (tmp_path / store.RESULTS_FILE).stat().st_size > 1.5 * big.nbytes        # the space stays this time
    assert sorted(p.name for p in tmp_path.iterdir()) == [store.RESULTS_FILE, store.RESULTS_FILE + '.lock']


def test_compact_gives_up_rather_than_drop_an_entry(tmp_path, monkeypatch):
    store.write_record(tmp_path, 'precious', {'answer': 42})
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [2]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()
    _flaky_copy(monkeypatch, 'precious', OSError('simulated'))
    messages = []
    with store.logging_to(messages.append):
        store.compact(tmp_path)
    monkeypatch.undo()
    assert path.read_bytes() == before                             # nothing committed: a byte copy gains nothing
    assert store.read_record(tmp_path, 'precious') == {'answer': 42}
    assert any('not rebuilt' in m for m in messages)
    store.compact(tmp_path)                                        # and it works once the copy does
    with h5py.File(path) as file:
        assert int(file.attrs.get(store.FREED_ATTRIBUTE, 0)) == 0
    assert store.read_record(tmp_path, 'precious') == {'answer': 42}


# Absent means absent (finding 2) -----------------------------------------------------------------------------

@pytest.mark.parametrize('error', [MemoryError('simulated'), OSError(errno.EIO, 'simulated EIO')])
def test_read_errors_not_about_the_stored_bytes_propagate_unchanged(tmp_path, monkeypatch, error):
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    store.write_record(tmp_path, 'r', {'b': 2})

    def failing(*args, **kwargs):
        raise error
    monkeypatch.setattr(store, '_decode_table', failing)
    with pytest.raises(type(error)) as raised:
        store.read_table(tmp_path, 't')
    assert raised.value is error and not isinstance(raised.value, FileNotFoundError)
    assert store.has_table(tmp_path, 't')                          # present: has_* does not decode
    monkeypatch.setattr(store.json, 'loads', failing)
    with pytest.raises(type(error)):
        store.read_record(tmp_path, 'r')


def test_a_damaged_entry_is_not_reported_missing(tmp_path):
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    _damage_entries(tmp_path / store.RESULTS_FILE, 'tables/t')
    try:
        store.read_table(tmp_path, 't')
    except FileNotFoundError:                                      # what callers of "missing" catch
        pytest.fail('a damaged entry was reported as missing')
    except store.DamagedEntry:
        pass
    assert store.has_table(tmp_path, 't')
    with pytest.raises(store.MissingEntry):                        # absent is still absent
        store.read_table(tmp_path, 'never_written')
    assert not store.has_table(tmp_path, 'never_written')


def test_cache_reads_are_misses_except_for_memory_errors(tmp_path, monkeypatch):
    store.write_cache_table(tmp_path, 't', pd.DataFrame({'a': [1]}))

    def no_memory(*args, **kwargs):
        raise MemoryError('simulated')

    def eio(*args, **kwargs):
        raise OSError(errno.EIO, 'simulated EIO')
    monkeypatch.setattr(store, '_decode_table', no_memory)
    with pytest.raises(MemoryError):
        store.read_cache_table(tmp_path, 't')
    monkeypatch.setattr(store, '_decode_table', eio)
    with store.logging_to(lambda message: None), pytest.raises(store.MissingEntry):
        store.read_cache_table(tmp_path, 't')                      # an error of the cache file: a miss
    monkeypatch.undo()
    assert store.read_cache_table(tmp_path, 't')['a'].tolist() == [1]


# A full disk (finding 3) -------------------------------------------------------------------------------------

def test_no_room_on_the_disk_refuses_to_start(tmp_path, monkeypatch):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    store.write_cache_record(tmp_path, 'c', {'k': 1})
    results, cache = tmp_path / store.RESULTS_FILE, tmp_path / store.CACHE_FILE
    before = results.read_bytes(), cache.read_bytes()
    real = os.statvfs

    def nearly_full(path):
        info = real(path)
        return os.statvfs_result((info.f_bsize, info.f_frsize, info.f_blocks, 10, 10, *tuple(info)[5:]))
    monkeypatch.setattr(store.os, 'statvfs', nearly_full)
    with pytest.raises(store.StoreError, match='full') as raised:
        store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))
    assert raised.value.errno == errno.ENOSPC
    messages = []
    with store.logging_to(messages.append):
        store.write_cache_arrays(tmp_path, 'z', x=np.ones(10))     # a miss, logged
    monkeypatch.undo()
    assert any('not written' in m and 'full' in m for m in messages), messages
    assert (results.read_bytes(), cache.read_bytes()) == before
    assert not list(tmp_path.glob('*.tmp-*'))
    store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))  # with room again, it writes


def test_room_is_checked_for_every_write_of_a_batch(tmp_path, monkeypatch):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    before = (tmp_path / store.RESULTS_FILE).read_bytes()
    with pytest.raises(store.StoreError, match='full'):
        with store.batch(tmp_path):
            store.write_record(tmp_path, 'first', {'v': 1})
            monkeypatch.setattr(store, '_no_room', lambda path, needed, final: 'the disk is (nearly) full: test')
            store.write_arrays(tmp_path, 'big', x=np.ones(1000))
    monkeypatch.undo()
    assert (tmp_path / store.RESULTS_FILE).read_bytes() == before and not list(tmp_path.glob('*.tmp-*'))


_FSIZE_CHILD = '''
    import os, resource, signal, sys
    import numpy as np
    from kappaccino.io import store
    folder, limit, target, check = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == 'check'
    if not check:
        store._no_room = lambda *args: None      # the check passes; the disk fills up after it
    store.set_log(lambda message: print('LOG', message[:160], flush=True))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
    x = np.random.default_rng(1).random(300_000)
    try:
        if target == 'results':
            with store.batch(folder):
                store.write_arrays(folder, 'new', x=x)
                store.write_record(folder, 'new_rec', {'n': 7})
        else:
            store.write_cache_arrays(folder, 'new', x=x)
        print('RESULT ok', flush=True)
    except BaseException as error:
        print('RESULT raised', type(error).__name__, str(error)[:200].replace(chr(10), ' '), flush=True)
'''


@pytest.mark.parametrize('target', ['results', 'cache'])
def test_a_file_size_limit_is_an_error_never_a_crash(tmp_path, target):
    """RLIMIT_FSIZE makes HDF5 writes fail as a full disk does.  With the room check, nothing starts; the process ends
    normally and the old file is intact."""
    for offset in (0, 4096, 1_000_000):
        folder = tmp_path / f'{target}{offset}'
        (store.write_record if target == 'results' else store.write_cache_record)(folder, 'old', {'v': 1})
        path = folder / (store.RESULTS_FILE if target == 'results' else store.CACHE_FILE)
        before = path.read_bytes()
        child = _python(_FSIZE_CHILD, folder, path.stat().st_size + offset, target, 'check')
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, (child.returncode, out, err[-2000:])
        if target == 'results':
            assert 'RESULT raised StoreError' in out and 'ulimit' in out, out
        else:
            assert 'RESULT ok' in out and 'not written' in out, out
        assert path.read_bytes() == before and not list(folder.glob('*.tmp-*'))


def test_a_write_that_fails_after_the_check_raises_and_keeps_the_old_version(tmp_path):
    """The disk fills up between the check and the write: the write is detected as failed (h5py only reports it from
    a destructor), raises StoreError, and results.h5 keeps its old version.  (The process may still crash later in
    HDF5; that is not guaranteed either way.)"""
    folder = tmp_path / 'results'
    store.write_record(folder, 'old', {'v': 1})
    path = folder / store.RESULTS_FILE
    before = path.read_bytes()
    child = _python(_FSIZE_CHILD, folder, path.stat().st_size + 20_000, 'results', 'nocheck')
    out, err = child.communicate(timeout=60)
    assert 'RESULT raised StoreError' in out and 'failed' in out, (child.returncode, out, err[-2000:])
    assert path.read_bytes() == before and not list(folder.glob('*.tmp-*'))
    assert store.read_record(folder, 'old') == {'v': 1}
    with pytest.raises(store.MissingEntry):
        store.read_arrays(folder, 'new')


# Batches fail as a whole (findings 4 and 5) ------------------------------------------------------------------

def test_an_inner_batch_that_fails_fails_the_outer_one(tmp_path):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()
    with pytest.raises(store.BatchFailed, match='inner batch failed'):
        with store.batch(tmp_path):
            store.write_record(tmp_path, 'outer', {'v': 1})
            try:
                with store.batch(tmp_path):
                    store.write_record(tmp_path, 'inner', {'v': 2})
                    raise RuntimeError('inner step failed')
            except RuntimeError:
                pass                                               # caught inside the outer batch
            with pytest.raises(store.BatchFailed):
                store.write_record(tmp_path, 'after', {'v': 3})   # a failed batch stays failed
    assert path.read_bytes() == before
    assert set(store.entries(tmp_path)[store.RECORDS]) == set()
    assert not list(tmp_path.glob('*.tmp-*'))


def test_a_failed_rebuild_inside_a_batch_neither_deadlocks_nor_commits(tmp_path, monkeypatch):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    store.write_table(tmp_path, 'broken', pd.DataFrame({'a': [2]}))
    store.write_record(tmp_path, 'also_broken', {'b': 3})
    _damage_entries(tmp_path / store.RESULTS_FILE, 'tables/broken', 'records/also_broken')
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()
    monkeypatch.setattr(store, 'WRITER_TIMEOUT', 20.0)
    with store.logging_to(lambda message: None), pytest.raises(store.BatchFailed):
        with store.batch(tmp_path):
            store.write_record(tmp_path, 'first', {'v': 1})
            with pytest.raises(store.StoreError, match='also_broken'):
                store.write_table(tmp_path, 'broken', pd.DataFrame({'a': [4]}))   # the rebuild fails
            # The lock and the temporary are released at once ...
            with open(tmp_path / (store.RESULTS_FILE + '.lock'), 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            assert not list(tmp_path.glob('*.tmp-*'))
            # ... and the next write says so at once instead of waiting for its own lock.
            start = time.monotonic()
            with pytest.raises(store.BatchFailed):
                store.write_record(tmp_path, 'second', {'v': 2})
            assert time.monotonic() - start < 2.0
    assert path.read_bytes() == before


# Links, stale temporaries, interrupts, lock holders (finding 6) ----------------------------------------------

def test_a_symlinked_results_file_is_followed(tmp_path):
    real_folder, stage = tmp_path / 'storage', tmp_path / 'stage'
    real_folder.mkdir()
    stage.mkdir()
    store.write_table(real_folder, 'kept', pd.DataFrame({'a': [1]}))
    link = stage / store.RESULTS_FILE
    link.symlink_to(real_folder / store.RESULTS_FILE)
    store.write_table(stage, 'new', pd.DataFrame({'b': [2]}))
    assert link.is_symlink()
    assert set(store.entries(real_folder)[store.TABLES]) == {'kept', 'new'}
    assert store.read_table(stage, 'new')['b'].tolist() == [2]


def test_a_temporary_of_a_reused_pid_is_removed_once_old(tmp_path):
    store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    young = tmp_path / f'{store.RESULTS_FILE}.tmp-{store._HOST}-{os.getppid()}-0000aaaa'
    old = tmp_path / f'{store.RESULTS_FILE}.tmp-{store._HOST}-{os.getppid()}-0000bbbb'
    for path in (young, old):
        path.write_bytes(b'partial')
    os.utime(old, (time.time() - 2 * store.STALE_AGE,) * 2)
    with store.logging_to(lambda message: None):
        store.write_table(tmp_path, 'u', pd.DataFrame({'a': [2]}))
    assert young.exists() and not old.exists()


def test_an_interrupt_during_the_commit_cleans_up(tmp_path, monkeypatch):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    path = tmp_path / store.RESULTS_FILE
    before = path.read_bytes()

    def interrupted(path):
        raise KeyboardInterrupt
    monkeypatch.setattr(store, '_fsync_file', interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))
    monkeypatch.undo()
    assert path.read_bytes() == before and not list(tmp_path.glob('*.tmp-*'))
    open_files = [h5py.h5f.FileID(i).name for i in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)]
    assert not any(b'.tmp-' in name for name in open_files), open_files
    with open(tmp_path / (store.RESULTS_FILE + '.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)           # released


def test_the_lock_file_names_its_holder_only_while_held(tmp_path):
    lock = tmp_path / (store.RESULTS_FILE + '.lock')
    seen = []
    real = store._put

    def spy(*args, **kwargs):
        seen.append(lock.read_text())
        return real(*args, **kwargs)
    store_put, store._put = store._put, spy
    try:
        store.write_table(tmp_path, 't', pd.DataFrame({'a': [1]}))
    finally:
        store._put = store_put
    assert seen and f'pid {os.getpid()} ' in seen[0]
    assert lock.read_text() == ''                                  # cleared when released
    assert 'not recorded' in store._holder(lock)
    lock.write_text('pid 12345 on elsewhere since 10:00:00\n')
    assert 'last holder' in store._holder(lock)


# The commit order, deterministically (finding 7) -------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='needs /proc to name the file of a descriptor')
def test_the_commit_closes_fsyncs_and_only_then_replaces(tmp_path, monkeypatch):
    store.write_table(tmp_path, 'kept', pd.DataFrame({'a': [1]}))
    events = []
    real_fsync, real_replace, real_close = os.fsync, os.replace, h5py.File.close

    def fsync(fd):
        events.append(('fsync', Path(os.readlink(f'/proc/self/fd/{fd}')).name))
        return real_fsync(fd)

    def replace(source, target):
        still_open = [h5py.h5f.FileID(i).name.decode() for i in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)]
        events.append(('replace', Path(source).name, Path(target).name,
                       any(Path(name).name == Path(source).name for name in still_open)))
        return real_replace(source, target)

    def close(self):
        if self.id.valid:
            events.append(('close', Path(self.filename).name))
        return real_close(self)
    monkeypatch.setattr(store.os, 'fsync', fsync)
    monkeypatch.setattr(store.os, 'replace', replace)
    monkeypatch.setattr(h5py.File, 'close', close)
    store.write_table(tmp_path, 'new', pd.DataFrame({'b': [2]}))
    monkeypatch.undo()
    temps = {e[1] for e in events if e[0] == 'close' and '.tmp-' in e[1]}
    assert len(temps) == 1, events
    temp = temps.pop()
    order = [e for e in events if e[0] != 'close' or e[1] == temp]
    assert order == [('close', temp), ('fsync', temp), ('replace', temp, store.RESULTS_FILE, False),
                     ('fsync', tmp_path.name)], events
