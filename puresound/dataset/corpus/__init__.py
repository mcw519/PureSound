"""Corpus preparation: scan audio into records, split it, and write metafiles.

Recipe-agnostic by construction. What differs between corpora -- where the
speaker hides in the path, what the filenames encode, which subsets exist -- is a
parameter or a small corpus module here, never a copy of this code under `egs/`.
"""

from .records import (
    METAFILE_HEADER,
    AudioRecord,
    read_inventory,
    read_metafile,
    tag_histogram,
    write_inventory,
    write_metafile,
)
from .resample import (
    ResampleReport,
    add_resample_arguments,
    build_resampled_tree,
    default_resample_root,
    mirrored_path,
    resample_if_requested,
)
from .scan import (
    DEFAULT_AUDIO_SUFFIXES,
    SPEAKER_ID_STRATEGIES,
    assert_disjoint,
    audio_info,
    drop_duplicate_files,
    iter_audio_files,
    parse_suffixes,
    sanitize_id,
    scan_folder,
    split_records,
)

__all__ = [
    "METAFILE_HEADER",
    "AudioRecord",
    "DEFAULT_AUDIO_SUFFIXES",
    "SPEAKER_ID_STRATEGIES",
    "ResampleReport",
    "assert_disjoint",
    "audio_info",
    "drop_duplicate_files",
    "add_resample_arguments",
    "build_resampled_tree",
    "default_resample_root",
    "resample_if_requested",
    "iter_audio_files",
    "mirrored_path",
    "parse_suffixes",
    "read_inventory",
    "read_metafile",
    "sanitize_id",
    "scan_folder",
    "split_records",
    "tag_histogram",
    "write_inventory",
    "write_metafile",
]
