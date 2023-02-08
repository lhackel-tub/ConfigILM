from os.path import join

from configvlm.extra.BEN_lmdb_utils import BENLMDBReader
from configvlm.extra.BEN_lmdb_utils import resolve_ben_data_dir

# an example that shows how the Reader is used to open the lmdb file and
# retrieve an image
if __name__ == "__main__":
    BEN_reader = BENLMDBReader(
        join(resolve_ben_data_dir(None), "BigEarthNetEncoded.lmdb"),
        (3, 120, 120),
        "RGB",
        label_type="old",
    )
    out = BEN_reader["S2A_MSIL2A_20170613T101031_0_48"]
    print(out)
    BEN_reader = BENLMDBReader(
        join(resolve_ben_data_dir(None), "BigEarthNetEncoded.lmdb"),
        (2, 120, 120),
        "S1",
        label_type="old",
    )
    out = BEN_reader["S2A_MSIL2A_20170613T101031_0_48"]
    print(out)
