from pathlib import Path
from railflaw_seg.catalog import Sample, cgan_group, nrsd_group, rsdds_group
from railflaw_seg.prepare import assign_splits


def test_derivative_group_keys():
    assert cgan_group('type_1_cropped_rail_10_lr_ud_var_3_0590') == rsdds_group(1, 'rail_10')
    assert nrsd_group('s0000505b_0_1') == nrsd_group('s0000505b_7_3')


def test_shared_images_and_source_groups_do_not_cross_splits(tmp_path):
    root = tmp_path
    samples = [Sample('A', str(i), root / f'{i}.jpg', root / f'{i}.png', 'binary', f'a:{i}') for i in range(30)]
    samples += [Sample('B', 'copy', root / 'copy.jpg', root / 'copy.png', 'binary', 'b:copy')]
    hashes = {'1.jpg': 'identical', 'copy.jpg': 'identical'}
    first = assign_splits(samples, hashes, root, 42, .8, .1)
    assert first == assign_splits(list(reversed(samples)), hashes, root, 42, .8, .1)
    assert first['a:1'] == first['b:copy']
    assert {value[1] for value in first.values()} == {'train', 'val', 'test'}
