"""Tests for utility functions."""

from stackone_defender.config import DEFAULT_RISKY_FIELDS
from stackone_defender.utils.boundary import (
    contains_boundary_patterns,
    generate_data_boundary,
    generate_xml_boundary,
    wrap_with_boundary,
)
from stackone_defender.utils.field_detection import (
    get_tool_override_fields,
    is_risky_field,
    matches_wildcard,
)
from stackone_defender.utils.structure import (
    create_size_metrics,
    detect_structure_type,
    estimate_size,
    is_paginated_response,
    is_plain_object,
)


class TestBoundary:
    def test_generate_data_boundary_format(self):
        b = generate_data_boundary()
        assert b.id
        assert b.start_tag.startswith("[UD-")
        assert b.end_tag.startswith("[/UD-")

    def test_generate_unique_ids(self):
        b1 = generate_data_boundary()
        b2 = generate_data_boundary()
        assert b1.id != b2.id

    def test_generate_xml_boundary(self):
        b = generate_xml_boundary()
        assert b.start_tag.startswith("<user-data-")
        assert b.end_tag.startswith("</user-data-")

    def test_wrap_with_boundary(self):
        b = generate_data_boundary()
        wrapped = wrap_with_boundary("hello", b)
        assert wrapped.startswith(b.start_tag)
        assert wrapped.endswith(b.end_tag)
        assert "hello" in wrapped

    def test_contains_boundary_patterns(self):
        assert contains_boundary_patterns("[UD-abc123]test[/UD-abc123]")
        assert contains_boundary_patterns("<user-data-xyz>test</user-data-xyz>")
        assert not contains_boundary_patterns("Hello world")


class TestFieldDetection:
    def test_is_risky_by_name(self):
        assert is_risky_field("name", DEFAULT_RISKY_FIELDS)
        assert is_risky_field("description", DEFAULT_RISKY_FIELDS)
        assert is_risky_field("content", DEFAULT_RISKY_FIELDS)
        assert is_risky_field("body", DEFAULT_RISKY_FIELDS)
        assert is_risky_field("subject", DEFAULT_RISKY_FIELDS)

    def test_is_not_risky(self):
        assert not is_risky_field("id", DEFAULT_RISKY_FIELDS)
        assert not is_risky_field("created_at", DEFAULT_RISKY_FIELDS)
        assert not is_risky_field("url", DEFAULT_RISKY_FIELDS)

    def test_is_risky_by_pattern(self):
        assert is_risky_field("first_name", DEFAULT_RISKY_FIELDS)
        assert is_risky_field("job_description", DEFAULT_RISKY_FIELDS)
        assert is_risky_field("email_body", DEFAULT_RISKY_FIELDS)

    def test_tool_overrides(self):
        assert is_risky_field("subject", DEFAULT_RISKY_FIELDS, "gmail_get_message")
        assert is_risky_field("snippet", DEFAULT_RISKY_FIELDS, "gmail_get_message")

    def test_matches_wildcard(self):
        assert matches_wildcard("gmail_get_message", "gmail_*")
        assert matches_wildcard("documents_list_files", "documents_*")
        assert not matches_wildcard("hris_list_employees", "gmail_*")

    def test_get_tool_override_fields(self):
        overrides = DEFAULT_RISKY_FIELDS.tool_overrides
        fields = get_tool_override_fields("gmail_get_message", overrides)
        assert fields is not None
        assert "subject" in fields

    def test_get_tool_override_fields_no_match(self):
        overrides = DEFAULT_RISKY_FIELDS.tool_overrides
        assert get_tool_override_fields("unknown_tool", overrides) is None


class TestStructure:
    def test_detect_array(self):
        assert detect_structure_type([1, 2, 3]) == "array"

    def test_detect_object(self):
        assert detect_structure_type({"name": "test"}) == "object"

    def test_detect_wrapped(self):
        assert detect_structure_type({"data": [1, 2]}) == "wrapped"
        assert detect_structure_type({"results": [1]}) == "wrapped"
        assert detect_structure_type({"items": []}) == "wrapped"

    def test_detect_primitive(self):
        assert detect_structure_type("hello") == "primitive"
        assert detect_structure_type(42) == "primitive"
        assert detect_structure_type(True) == "primitive"

    def test_detect_null(self):
        assert detect_structure_type(None) == "null"

    def test_is_plain_object(self):
        assert is_plain_object({"key": "value"})
        assert not is_plain_object([1, 2])
        assert not is_plain_object(None)
        assert not is_plain_object("string")
        assert not is_plain_object(42)

    def test_is_paginated_response(self):
        assert is_paginated_response({"data": [1, 2], "next": "cursor123"})
        assert is_paginated_response({"results": [1], "total": 100})
        assert not is_paginated_response({"data": [1, 2]})
        assert not is_paginated_response({"name": "test"})
        assert not is_paginated_response([1, 2])

    def test_estimate_size(self):
        assert estimate_size(None) == 4
        assert estimate_size("hello") == 7  # 5 + 2 quotes
        assert estimate_size(42) == 2
        assert estimate_size(True) == 4
        assert estimate_size(False) == 5

    def test_create_size_metrics(self):
        m = create_size_metrics()
        assert m.estimated_bytes == 0
        assert m.string_count == 0
        assert not m.size_limit_hit
