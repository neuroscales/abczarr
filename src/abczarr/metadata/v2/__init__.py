from abczarr._core.imports import import_all

import_all(
    (
        ".array",
        ".base",
        ".codecs",
        ".dtypes",
        ".filters",
    ),
    locals(),
    __package__,
    add_to_all="attrs"
)
