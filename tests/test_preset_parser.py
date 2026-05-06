from pathlib import Path

from gulls_pipeline.presets.parser import parse_prm_file, parse_prm_text


def test_parse_key_values_strips_comments_and_whitespace():
    prm = parse_prm_text(
        """
        # comment-only line
        RUN_NAME = smoke_general   # inline comment
        OUTPUT_DIR= /tmp/out
        EMPTY_VALUE =
        VALUE_WITH_EQUALS = alpha=beta
        malformed line without separator
        """
    )

    assert prm.key_order == (
        "RUN_NAME",
        "OUTPUT_DIR",
        "EMPTY_VALUE",
        "VALUE_WITH_EQUALS",
    )
    assert prm.settings == {
        "RUN_NAME": "smoke_general",
        "OUTPUT_DIR": "/tmp/out",
        "EMPTY_VALUE": "",
        "VALUE_WITH_EQUALS": "alpha=beta",
    }


def test_duplicate_keys_use_last_value_and_are_recorded():
    prm = parse_prm_text(
        """
        RUN_NAME=first
        OUTPUT_DIR=/first
        RUN_NAME=second
        RUN_NAME=third
        """
    )

    assert prm.key_order == ("RUN_NAME", "OUTPUT_DIR")
    assert prm.settings["RUN_NAME"] == "third"
    assert [entry.value for entry in prm.duplicates["RUN_NAME"]] == ["second", "third"]
    assert [entry.line_number for entry in prm.duplicates["RUN_NAME"]] == [4, 5]


def test_to_text_preserves_order_applies_overrides_and_appends_new_keys():
    prm = parse_prm_text(
        """
        RUN_NAME=first
        OUTPUT_DIR=/first
        RUN_NAME=second
        NSUBRUNS=1
        """
    )

    assert prm.to_text(
        {
            "NSUBRUNS": 3,
            "NEW_SETTING": "enabled",
            "RUN_NAME": "override",
        }
    ) == (
        "RUN_NAME=override\n"
        "OUTPUT_DIR=/first\n"
        "NSUBRUNS=3\n"
        "NEW_SETTING=enabled\n"
    )


def test_parse_prm_file_reads_utf8_file(tmp_path: Path):
    path = tmp_path / "preset.prm"
    path.write_text("RUN_NAME=file\n# ignored\nNSUBRUNS=2\n", encoding="utf-8")

    prm = parse_prm_file(path)

    assert prm.settings == {"RUN_NAME": "file", "NSUBRUNS": "2"}
    assert prm.to_text() == "RUN_NAME=file\nNSUBRUNS=2\n"
